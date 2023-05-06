from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
import os
import tempfile
import requests
from abc import ABC
from pathlib import Path
from typing import List
from urllib.parse import urlparse
import os
import logging
logger = logging.getLogger(__file__)


class BasePDFLoader(BaseLoader, ABC):
    """Base loader class for PDF files.

    Defaults to check for local file, but if the file is a web path, it will download it
    to a temporary file, and use that, then clean up the temporary file after completion
    """

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
        self.web_path = None
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

        # If the file is a web path, download it to a temporary file, and use that
        if not os.path.isfile(self.file_path) and self._is_valid_url(self.file_path):
            r = requests.get(self.file_path)

            if r.status_code != 200:
                raise ValueError(
                    "Check the url of your file; returned status code %s"
                    % r.status_code
                )

            self.web_path = self.file_path
            self.temp_file = tempfile.NamedTemporaryFile()
            self.temp_file.write(r.content)
            self.file_path = self.temp_file.name
        elif not os.path.isfile(self.file_path):
            raise ValueError(
                "File path %s is not a valid file or url" % self.file_path)

    def __del__(self) -> None:
        if hasattr(self, "temp_file"):
            self.temp_file.close()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @property
    def source(self) -> str:
        return self.web_path if self.web_path is not None else self.file_path


class PyPDFLoader(BasePDFLoader):
    """Loads a PDF with pypdf and chunks at character level.

    Loader also stores page numbers in metadatas.
    """

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            import pypdf
        except ImportError:
            raise ValueError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        import pypdf

        with open(self.file_path, "rb") as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj)
            sourcelink = pdf_reader.metadata.get("/Sourcelink")
            return [
                Document(
                    page_content=page.extract_text(),
                    metadata={"source": self.file_path,
                              "page": i, "sourcelink": sourcelink},
                )
                for i, page in enumerate(pdf_reader.pages)
            ]


class CustomPyPDFDirectoryLoader(BaseLoader):
    """Loads a directory with PDF files with pypdf and chunks at character level.

    Loader also stores page numbers in metadatas.
    """

    def __init__(
        self,
        path: str,
        glob: str = "**/[!.]*.pdf",
        silent_errors: bool = False,
        load_hidden: bool = False,
        recursive: bool = False,
    ):
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.silent_errors = silent_errors

    @staticmethod
    def _is_visible(path: Path) -> bool:
        return not any(part.startswith(".") for part in path.parts)

    def load(self) -> List[Document]:
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        loader = PyPDFLoader(str(i))
                        sub_docs = loader.load()
                        for doc in sub_docs:
                            doc.metadata["source"] = str(i)
                        docs.extend(sub_docs)
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(e)
                        else:
                            raise e

        return docs
