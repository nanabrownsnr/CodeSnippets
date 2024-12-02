import os
from openai import OpenAI

def use_grok(system,user):
    XAI_API_KEY = os.getenv("XAI_API_KEY")
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


def check_customer_understanding(user_input):
    system_input = "You are an analyst with a strong understanding of MiFID II, its definition, regulations, who it affects, and purpose. Read the text I have submitted and check if the consumer (buyer) clearly articulated their understanding of product and process on the phone when talking to the customer sales agent. Respond with either yes or no"
    response = use_grok(system_input,user_input)
    print(response)
    return response




def generate_youtube_description(user_input):
    system_input = "Generate a youtube description for this story."
    response = use_grok(system_input,user_input)
    print(response)
    return response


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
    print(response)
    return response
