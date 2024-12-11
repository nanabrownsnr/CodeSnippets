import os
from openai import OpenAI

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

        Ensure all information provided to the customer is fair, clear, and not misleading.
        Disclose the identity of the seller and the purpose of the call at the outset.
        Provide detailed information about the product or service, including its characteristics, costs, risks, and benefits.
        Explain the terms of the contract in a way that the customer can understand. 
        Record phone conversations or electronic communications involving client orders to ensure transparency and legal certainty.Act in the best interest of the client, ensuring that the product or service meets their needs.
        Obtain clear and explicit consent from the customer before finalizing the transaction. 
        Inform the customer of their right to withdraw from the agreement within a specified cooling-off period. 
        Maintain the confidentiality of customer data throughout the transaction process.
        Provide a transparent breakdown of costs, including any commissions or additional charges.
        Avoid any hidden fees or costs that may mislead the customer.
        Inform customers about the process for submitting complaints and the timeline for resolution.
        Refrain from using aggressive sales tactics or pressuring the customer into making an immediate decision.
        
        Read the transcript of the the sales call submitted and determine if the call was complaint with these obligations. 
        If yes, respond with yes, if not respond with no followed by the list of obligations that were not met.
        """
    response = use_grok(system_input,user_input)
    print(response)
    return response
