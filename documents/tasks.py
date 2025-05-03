from huey.contrib.djhuey import task

from documents.methods import process_ocr, process_vector, summarize_document
from documents.models import Document


@task()
def process_document(document: Document):
    ocr_content = process_ocr(document)
    summarize_document(document, ocr_content)
    process_vector(document, ocr_content)
