from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import pandas as pd
import csv

model = M2M100ForConditionalGeneration.from_pretrained("m2m100_small_fr_sba")
tokenizer = M2M100Tokenizer.from_pretrained("m2m100_small_fr_sba")

final=[]
data=pd.read_csv('output-3.csv')
for i in range (len(data['sentences'])):
    text_to_translate=data['sentences'][i]

  
    #text_to_translate = data['sentences']



   
    model_inputs = tokenizer(text_to_translate, return_tensors="pt")

    # translate to French
    #from main import model
    gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("wo"))
    output=tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    final.append(output)
    #print(final)
    

    # Function to write lists to a CSV file
data = final

# Open the CSV file for writing
with open('sba.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Write the column header
    writer.writerow(['sba'])

    # Write each list to a separate row in the CSV file
    for row in data:
        writer.writerow(row)
    
