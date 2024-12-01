import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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
    return_timestamps=True
)


def speech_to_text(audio_file_url):
    result = pipe(audio_file_url)
    return result["text"]

# output = "You have reached the essence of Oregon. This is Donna. I'll be assisting you with your inquiries today. Please be informed that this call is being recorded and monitored for quality assurance purposes. How may I help you? Well, I got this essence of Oregon oil for shipping and handling costs of $5.99, a sample of it. and if I want to cancel the order, I had to do it within 15 days. And so that is what I wanted to do. I didn't want to get, you know, like a monthly for, what is it, $83 a month. Okay. I can't afford that. Okay, I'm more than happy. I'm happy to assist you. So for me to be able to pull up your subscription here, could you kind of provide me your first and your last name? Carolyn, C-A-R-O-L-Y-N, Lake, L-A-K-E. O-L-Y-N. And then Lake. Yes. L-A-K-E. Yes. Okay. Let's just go ahead to pull up your subscription here or your account. Okay. And could you kind of verify your email address, please? It's lake3921 at hotmail.com. How about your shipping address? 310 Warren Avenue, number 2, Gillette, Wyoming, 82716. And is your shipping address the same as your billing address? Correct. How about your phone number? 307-680-2068. Okay, thank you very much for that information, Ms. Lake. If you don't want me to ask, you may know the reason why you want to cancel the subscription. Well, I thought I was just getting a sample order of it. You know, always curious to how it worked and everything. Yes, I mean, have you already used it? Not all of it, but I have. Okay. I have been using it, yes. Okay, I do understand that, Ms. Lake. This is what I can do for you, for you to be able to maximize or enjoy the benefits of the organ. I'm going to extend your pretrial for another 15 days with no charge, So at least you do have 15 days to enjoy the amazing product. And then give us a call back before the end of that 15-day additional or extension period to give us the feedback. Because what I heard from you is that you haven't used the product that much. So you're not, you know, you did not yet get the benefits of it. So that's the reason why I'm extending your period or your pretrial for you to be able to enjoy and discover the benefits of Essence of Argan. Okay? Okay. What date would that be? Okay, let me check here. So you're already extended your pretrial. It will end on August 24th. So you need to give us a call back before August 24th to give us a feedback. I mean, what happened to the product, if it didn't something, you know, it gives you the benefits that you need. But let's say you love the product, you like the product, you don't need to give us a call back then. It will be automatically, you will be receiving another bottle for your monthly subscription. And on set of a call, I heard that you're not, you know, you're dealing about the price, which is $83.86, right? Okay. So what I can do for you is, aside from extending your pretrial for another 15 days, I'm also giving you my employee discount, which is 20% discount. So instead of paying $83.80, you're going to pay only $67.04. How about that? I can't. I can't afford that. Okay. This is my maximum employee discount, a 40% employee discount. I'll give it to you. It's actually a lifetime employee discount. So instead of paying $83.80, you'll just be paying $50.28. No, I can't. I cannot do that. I can't. I'd like to. And I wanted to ask you, too. Go ahead. How often am I supposed to use that? Like at night or? Yes. As long as you want, ma'am. Just use it at night? Yes. As long as you want. As long as you need it. Oh. Okay. Okay, Ms. Lake, can I just put you on for a minute? Yeah, I can't afford. Can I just put you on for a minute? Okay. Sure. Thank you. Thank you. Thank you very much for your patience waiting, Ms. Lake. Okay. I want to talk to my supervisor, and she asked me, what amount would you feel the product would be more affordable for you and make your continue your subscription with us? oh um if we will ask you do you think i could i could you know like 15 dollars oh i know that's you know but that okay ma'am don't worry that's great uh because i've already given you my employee discount of 40 percent right so my supervisor can give you another 40 So that's an equivalent of 80%. So instead of paying $83.80, what you'll be paying is only $16.76 every month for everybody. How about that? That sounds great. Okay, that's great. So I'm extending your pretrial until August 24th. So after August 24th, you will be charged only $16.76. Okay? So every month you will be paying, I mean, you will be receiving a bottle of essence of argon, and you will be enjoying the benefits of the product, but you will be only paying $16.76. Okay? Okay. Okay, then. So that's great. So just for a recap, Ms. Lake, I extended your pretrial for another 15 days, and it's going to end on August 24th. So you can give us a call back within that period of time to give us the feedback. But like what I've said, if you enjoy the product, it gives you the benefits, the amazing benefits of the product, you don't need to give us a call back. It will be automatically. You will be receiving a subscription every month, but you will be only paying $16.76 because we already give you our employee discount of 80% for lifetime. Okay? Okay. All right. Okay, then. So is there anything else we can help you with? No, that would be it. Okay, then. So we value your business. Thank you for calling Essence of Oregon. Have a great day. You too. Bye-bye."
