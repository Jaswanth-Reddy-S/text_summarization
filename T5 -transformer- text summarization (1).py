
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

#model = T5ForConditionalGeneration.from_pretrained('t5-small')
#tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

text="Want to remove both the ugly dirt you see and kill 99.9 Percent of bacteria you don't? Try Cif Antibac & Shine Multipurpose Disinfectant wipes which clean & disinfect your home, eliminating 99.9 Percent of bacteria and cleaning at the same time. Cif Antibac & Shine multipurpose wipes are safe for cleaning & disinfecting virtually all surfaces** and are tough on dirt & scum found in the home: From sinks to stainless steel, cookers to bins, toilets to tables and plenty more in-between. Now you can enjoy hygienic, sparkling clean surfaces all around your home! Cif antibacterial wipes are a convenient solution to clean the visible and invisible dirt all around your home more sustainably than with a regular wipe, as they are made with 100 Percent biodegradable plant fibres. Cif Antibac & Shine Multipurpose wipes deliver beautiful cleaning results and leave behind a fresh fragrance. For beautiful, hygienic results, break the seal and open by peeling the sticker halfway back then wipe your surfaces to remove dirt – no need to rinse. Make sure the pack is resealed after use to avoid wipes drying out. Do not flush down the toilet. Discover more beautiful cleaning products, tips & tricks from our online community by joining the CifSquad – full details on our website. Through the Unilever Sustainable Living Plan, Unilever’s mission is to make sustainable living commonplace. By 2025 we have committed to ensuring that all our plastic packaging is fully reusable, recyclable or compostable.Eliminates bacteria like Salmonella, E.coli and Staphylococcus aureus (MRSA)"

preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,#no_repeat_ngram_size=2 so that no 2-gram appears twice:
                                    min_length=150,
                                    max_length=1200,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)




