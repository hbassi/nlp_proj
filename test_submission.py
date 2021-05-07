import json, sys
from data_processing import tokenize
import torch
from model_loader import load_model

'''
MAKE SURE DEVICE IS SET CORRECTLY IN EVERY FILE.
'''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Hyperparameters
'''
MAXLENGTH = 185

def eval(text):
	# This is where you call your model to get the number of stars output
	tokenText = tokenize(text, MAXLENGTH)
	inputIds = tokenText['input_ids'].to(DEVICE)
	attentionMask = tokenText['attention_mask'].to(DEVICE)
	
	#Update this in order to change model to use
	model = load_model('bert')
	
	outputTensor = model(inputIds, attentionMask)
	rating = torch.argmax(outputTensor).item()
	return rating + 1

'''
Takes in command line arguement for json files. Will write outputs of eval() into output.jsonl
'''
if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			for line in fr:
				review = json.loads(line)
				fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")