#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

text ="Every second counts, whether you’re running late or running to beat your personal best. Nothing should distract you from your focus – especially not sweat. Sure Men Sensitive Anti-perspirant Deodorant Stick 50 ml is formulated to provide 48-hour protection against sweat and odour, so you can feel fresh, dry and protected all day long, and always prepared for whatever happens. This deodorant stick also contains Sure’s innovative Motionsense technology. It works like this: unique microcapsules sit on the surface of your skin. When you move, friction breaks those microcapsules and they release more fragrance. So every time you move, Motionsense keeps you fresh and free from odour. Whether you’re working hard in the office, playing hard out and about or just meeting up with friends, Sure Men Sensitive anti-perspirant will give you all the back-up you need. Sure, it won't let you down. So, get all day freshness and 48-hour protection morning to night with Sure Men Sensitive anti-perspirant deodorant to keep sweat and odour at bay. How to use: Apply your Sure anti-perspirant deodorant stick onto dry underarms evenly to release a reassuring masculine scent with sensual notes of sandalwood, patchouli and vanilla. Avoid contact with eyes and broken skin."


preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=1000,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)

# Summarized output from above ::::::::::
# the us has over 637,000 confirmed Covid-19 cases and over 30,826 deaths. 
# president Donald Trump predicts some states will reopen the country in april, he said. 
# "we'll be the comeback kids, all of us," the president says.


# link- https://towardsdatascience.com/simple-abstractive-text-summarization-with-pretrained-t5-text-to-text-transfer-transformer-10f6d602c426

# In[28]:


import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

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
                                    min_length=150,
                                    max_length=1200,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)


# In[1]:


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
                                    min_length=180,
                                    max_length=1200,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)


# In[ ]:


text="Surf Lavender & Spring Jasmine is the perfect combination of two iconic scents. Lavender's floral, sweet scent combined with its woody, pungent perfume is both stimulating and soothing and has been said to be used to freshen laundry since Roman times. Spring Jasmine's warm, comforting and sweet scent is fabled for its aphrodisiac qualities. Called the “Queen of the Night" in India, its fragrance is said to produce feelings of optimism and playfulness. Surf Lavender & Spring Jasmine with its soothing fragrance is available in washing liquid, washing powder and washing capsules and is suitable for washing both colours and whites. Surf's laundry range brings you the joy of fragrance, long after you've washed your clothes. With burst after burst of uplifting fragrance released right through your day, your laundry stays fragrantly fresh, with a brilliant deep clean you'll love. Surf powder offers brilliant cleaning and excellent fragrance, with outstanding results even in cold water. It has brilliant cleaning power and is great on white clothes. To use Surf powder effectively, add it to the dispensing drawer of your washing machine, put in the laundry and start the wash. For the best results in soft water, use 50ml Surf powder for light loads, 90ml for standard loads and 170ml for larger or dirtier loads. In medium water, use 90ml for light loads, 130ml for standard and 201ml maximum. In hard water, use 130ml for light, 170 ml for standard and 250ml maximum. (1 hour cycle)"


# In[4]:


import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

#model = T5ForConditionalGeneration.from_pretrained('t5-small')
#tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

text="Surf Lavender & Spring Jasmine is the perfect combination of two iconic scents. Lavender's floral, sweet scent combined with its woody, pungent perfume is both stimulating and soothing and has been said to be used to freshen laundry since Roman times. Spring Jasmine's warm, comforting and sweet scent is fabled for itsaphrodisiac qualities. Called the “Queen of the Night in India, its fragrance is said to produce feelings of optimism and playfulness. Surf Lavender & Spring Jasmine with its soothing fragrance is available in washing liquid, washing powder and washing capsules and is suitable for washing both colours and whites.Surf's laundry range brings you the joy of fragrance, long after you've washed your clothes. With burst after burst of uplifting fragrance released right through your day, your laundry stays fragrantly fresh, with a brilliant deep clean you'll love. Surf powder offers brilliant cleaning and excellent fragrance, with outstanding results even in cold water. It has brilliant cleaning power and is great on white clothes. To use Surf powder effectively, add it to the dispensing drawer of your washing machine, put in the laundry and start the wash. For the best results in soft water, use 50ml Surf powder for light loads, 90ml for standard loads and 170ml for larger or dirtier loads. In medium water, use 90ml for light loads, 130ml for standard and 201ml maximum. In hard water, use 130ml for light, 170 ml for standard and 250ml maximum. (1 hour cycle)"

preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=5,
                                    no_repeat_ngram_size=2,#no_repeat_ngram_size=2 so that no 2-gram appears twice:
                                    min_length=180,
                                    max_length=1200,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)


# In[11]:


import torch
import json 
#from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

#model = T5ForConditionalGeneration.from_pretrained('t5-small')
#tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")

model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")

text="Surf Lavender & Spring Jasmine is the perfect combination of two iconic scents. Lavender's floral, sweet scent combined with its woody, pungent perfume is both stimulating and soothing and has been said to be used to freshen laundry since Roman times. Spring Jasmine's warm, comforting and sweet scent is fabled for itsaphrodisiac qualities. Called the “Queen of the Night in India, its fragrance is said to produce feelings of optimism and playfulness. Surf Lavender & Spring Jasmine with its soothing fragrance is available in washing liquid, washing powder and washing capsules and is suitable for washing both colours and whites.Surf's laundry range brings you the joy of fragrance, long after you've washed your clothes. With burst after burst of uplifting fragrance released right through your day, your laundry stays fragrantly fresh, with a brilliant deep clean you'll love. Surf powder offers brilliant cleaning and excellent fragrance, with outstanding results even in cold water. It has brilliant cleaning power and is great on white clothes. To use Surf powder effectively, add it to the dispensing drawer of your washing machine, put in the laundry and start the wash. For the best results in soft water, use 50ml Surf powder for light loads, 90ml for standard loads and 170ml for larger or dirtier loads. In medium water, use 90ml for light loads, 130ml for standard and 201ml maximum. In hard water, use 130ml for light, 170 ml for standard and 250ml maximum. (1 hour cycle)"

preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=5,
                                    no_repeat_ngram_size=2,#no_repeat_ngram_size=2 so that no 2-gram appears twice:
                                    min_length=180,
                                    max_length=1200,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)


# In[13]:


# Pegasus-xsum not good


# In[12]:



import torch
import json 
#from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

#model = T5ForConditionalGeneration.from_pretrained('t5-small')
#tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

text="Surf Lavender & Spring Jasmine is the perfect combination of two iconic scents. Lavender's floral, sweet scent combined with its woody, pungent perfume is both stimulating and soothing and has been said to be used to freshen laundry since Roman times. Spring Jasmine's warm, comforting and sweet scent is fabled for itsaphrodisiac qualities. Called the “Queen of the Night in India, its fragrance is said to produce feelings of optimism and playfulness. Surf Lavender & Spring Jasmine with its soothing fragrance is available in washing liquid, washing powder and washing capsules and is suitable for washing both colours and whites.Surf's laundry range brings you the joy of fragrance, long after you've washed your clothes. With burst after burst of uplifting fragrance released right through your day, your laundry stays fragrantly fresh, with a brilliant deep clean you'll love. Surf powder offers brilliant cleaning and excellent fragrance, with outstanding results even in cold water. It has brilliant cleaning power and is great on white clothes. To use Surf powder effectively, add it to the dispensing drawer of your washing machine, put in the laundry and start the wash. For the best results in soft water, use 50ml Surf powder for light loads, 90ml for standard loads and 170ml for larger or dirtier loads. In medium water, use 90ml for light loads, 130ml for standard and 201ml maximum. In hard water, use 130ml for light, 170 ml for standard and 250ml maximum. (1 hour cycle)"

preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=5,
                                    no_repeat_ngram_size=2,#no_repeat_ngram_size=2 so that no 2-gram appears twice:
                                    min_length=180,
                                    max_length=1200,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)


# In[36]:



        
a=''
for x in output:
    if x=='.':
        print('*'+a+'.')
        a=''
    else:
        a=a+x        


# In[37]:



import torch
import json 
#from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

#model = T5ForConditionalGeneration.from_pretrained('t5-small')
#tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large")

text="Surf Lavender & Spring Jasmine is the perfect combination of two iconic scents. Lavender's floral, sweet scent combined with its woody, pungent perfume is both stimulating and soothing and has been said to be used to freshen laundry since Roman times. Spring Jasmine's warm, comforting and sweet scent is fabled for itsaphrodisiac qualities. Called the “Queen of the Night in India, its fragrance is said to produce feelings of optimism and playfulness. Surf Lavender & Spring Jasmine with its soothing fragrance is available in washing liquid, washing powder and washing capsules and is suitable for washing both colours and whites.Surf's laundry range brings you the joy of fragrance, long after you've washed your clothes. With burst after burst of uplifting fragrance released right through your day, your laundry stays fragrantly fresh, with a brilliant deep clean you'll love. Surf powder offers brilliant cleaning and excellent fragrance, with outstanding results even in cold water. It has brilliant cleaning power and is great on white clothes. To use Surf powder effectively, add it to the dispensing drawer of your washing machine, put in the laundry and start the wash. For the best results in soft water, use 50ml Surf powder for light loads, 90ml for standard loads and 170ml for larger or dirtier loads. In medium water, use 90ml for light loads, 130ml for standard and 201ml maximum. In hard water, use 130ml for light, 170 ml for standard and 250ml maximum. (1 hour cycle)"

preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=5,
                                    no_repeat_ngram_size=2,#no_repeat_ngram_size=2 so that no 2-gram appears twice:
                                    min_length=180,
                                    max_length=1200,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)
print("========================")
a=''
for x in output:
    if x=='.':
        print('*'+a+'.')
        a=''
    else:
        a=a+x  


# In[ ]:




