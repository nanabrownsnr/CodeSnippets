import os
from fastapi import FastAPI, UploadFile

import uvicorn
from fastapi import FastAPI, File, UploadFile

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from openai import OpenAI




####################################################################################################
# Configuration
####################################################################################################

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
)

def use_grok(system,user):
    XAI_API_KEY = os.getenv("XAI_API_KEY","xai-SsjTNF2zhTdoDM3jwdPHc62NHD6WQBzIjCMkS8oDl8Ec8hLVjAN2GlOWxX5FzVGRnrPrF1VVFEHf6MDO")
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    completion = client.chat.completions.create(
        model="grok-beta",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return completion.choices[0].message.content


def check_compliance(user_input):
    system_input = """
        
        Here is a comprehensive list of obligations that must be adhered to when selling products or services to a customer via phone, as derived from the attached document (Directive 2014/65/EU):

        Clear and Transparent Communication:

        Ensure all information provided to the customer is fair, clear, and not misleading.
        Disclose the identity of the seller and the purpose of the call at the outset.
        Provision of Comprehensive Information:

        Provide detailed information about the product or service, including its characteristics, costs, risks, and benefits.
        Explain the terms of the contract in a way that the customer can understand.
        Recording of Communications:

        Record phone conversations or electronic communications involving client orders to ensure transparency and legal certainty.
        Retain records in a durable medium for regulatory and evidential purposes.
        Client Protection Measures:

        Act in the best interest of the client, ensuring that the product or service meets their needs.
        Perform a suitability or appropriateness assessment based on the customer’s profile and requirements.
        Customer Consent and Right to Withdraw:

        Obtain clear and explicit consent from the customer before finalizing the transaction.
        Inform the customer of their right to withdraw from the agreement within a specified cooling-off period.
        Data Protection and Confidentiality:

        Adhere to data protection laws, ensuring customer information is used solely for the purpose intended.
        Maintain the confidentiality of customer data throughout the transaction process.
        Obligations on Pricing and Fees:

        Provide a transparent breakdown of costs, including any commissions or additional charges.
        Avoid any hidden fees or costs that may mislead the customer.
        Handling Complaints:

        Establish an accessible and effective procedure for handling customer complaints.
        Inform customers about the process for submitting complaints and the timeline for resolution.
        Avoidance of High-Pressure Tactics:

        Refrain from using aggressive sales tactics or pressuring the customer into making an immediate decision.
        Allow the customer sufficient time to consider the offer and seek additional advice if needed.
        
        Read the transcript of the the sales call submitted and determine if the call was complaint with these obligations. 
        If yes, respond with yes, if not respond with no followed by the list of obligations that were not met.
        """
    response = use_grok(system_input,user_input)
    return response


####################################################################################################
# Server API 
####################################################################################################

app = FastAPI()


@app.get('/health')
def api_health():
    return {"status":200,"message":"running ok"}

@app.post("/speech-to-text/")
def speech_to_text(audio_file: UploadFile = File(...)) -> dict:
    """
    Transcribe speech from an uploaded audio file.
    
    Args:
        audio_file (UploadFile): The uploaded audio file.
        
    Returns:
        dict: Transcription result with text and optional timestamps.
    """
    try:
        # Read audio content
        audio_data = audio_file.read()
        
        # Save to temporary file
        temp_file = f"./{audio_file.filename}"
        with open(temp_file, "wb") as f:
            f.write(audio_data)
        
        # Perform transcription
        result = pipe(temp_file)
        return {"text": result["text"], "timestamps": result.get("chunks", [])}
    except Exception as e:
        return {"error": str(e)}
    
    
@app.post("/generate_message/")
def getMessage(information:str):
    return check_compliance(information)


####################################################################################################
# Program Entry Point
####################################################################################################

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8500))
    uvicorn.run(
        "api:app",  # Replace "main" with the module name where your app is defined
        host="0.0.0.0",
        port= 8500,  # Replace port with your desired port number
        reload=True  # Optional: Enables automatic reloading on file changes
    )