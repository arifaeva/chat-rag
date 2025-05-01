from huey.contrib.djhuey import task

from documents.models import DOC_STATUS_COMPLETE, Document
from core.ai.mistral import mistral
from core.ai.prompt_manager import PromptManager

@task()
def process_document(document: Document):
        uploaded_pdf = mistral.files.upload(
            file={
                "file_name": document.name,
                "content": open(f"media/{document.file.name}", "rb")
            },
            purpose="ocr"
        ) # Private

        signed_url = mistral.files.get_signed_url(file_id=uploaded_pdf.id)

        ocr_result = mistral.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url
            }
        )

        content = ""

        for page in ocr_result.dict().get("pages", []):
                content += page["markdown"]

        pm = PromptManager(model="gpt-4.1")
        pm.add_message("system", "Please summarize the provided text. Extract also the key points.")
        pm.add_message("user", f"Content : {content}")

        summarized_content = pm.generate()

        document.raw_text = content
        document.summary = summarized_content
        document.status = DOC_STATUS_COMPLETE
        document.save()