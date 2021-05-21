import torch

#model = T5ForConditionalGeneration.from_pretrained('t5-small')
#tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

text ="Every second counts, whether you’re running late or running to beat your personal best. Nothing should distract you from your focus – especially not sweat. Sure Men Sensitive Anti-perspirant Deodorant Stick 50 ml is formulated to provide 48-hour protection against sweat and odour, so you can feel fresh, dry and protected all day long, and always prepared for whatever happens. This deodorant stick also contains Sure’s innovative Motionsense technology. It works like this: unique microcapsules sit on the surface of your skin. When you move, friction breaks those microcapsules and they release more fragrance. So every time you move, Motionsense keeps you fresh and free from odour. Whether you’re working hard in the office, playing hard out and about or just meeting up with friends, Sure Men Sensitive anti-perspirant will give you all the back-up you need. Sure, it won't let you down. So, get all day freshness and 48-hour protection morning to night with Sure Men Sensitive anti-perspirant deodorant to keep sweat and odour at bay. How to use: Apply your Sure anti-perspirant deodorant stick onto dry underarms evenly to release a reassuring masculine scent with sensual notes of sandalwood, patchouli and vanilla. Avoid contact with eyes and broken skin."


preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,#no_repeat_ngram_size=2 so that no 2-gram appears twice:
                                    min_length=180,
                                    max_length=1200,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)
