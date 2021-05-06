import pandas as pd
import nlpaug
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import sys

class DataAugmentation:
    #{aug} {
    # 1: run on colab
    # 2: run on laptop
    # 3: run on ray's laptop
    # NOTE: DECIDED TO ONLY RUN ON THE LAPTOPS NO COLAB.
    # }
    def __init__(self, aug_num):
        self.PATH = None
        self.aug_num = aug_num
        self.data = pd.read_json('yelp_review_training_dataset.jsonl',lines=True)
        if self.aug_num == '1':
            self.data = pd.read_json('/content/gdrive/MyDrive/nlp_proj/yelp_review_training_dataset.jsonl',lines=True)
            self.PATH = '/content/gdrive/MyDrive/nlp_proj/augmentations/'
        elif self.aug_num == '2':
            print("me")
            self.PATH = 'C:/Users/Hardeep/Desktop/nlp_proj/augmentations'
        elif self.aug_num == '3':
            self.PATH = '/Users/raymondwang/desktop/cs182/nlp_proj/augmentations/'
        else:
            print('The number entered does not correlate to a path')
        print("Data has been read into environment and path has been set...")

    def run(self):
        X = self.data['text']
        y = self.data['stars']
        if self.aug_num == '2':

            print("Running first augmentation...")
            aug_syn = naw.SynonymAug(aug_src='wordnet')
            augmented_text2 = X.apply(aug_syn.augment)
            df = pd.DataFrame()
            df['text'] = augmented_text2
            df['stars'] = y
            df.to_csv(self.PATH + 'synonym_aug.csv', index=False)
            
            # print("Running second augmentation...")
            # aug_rws = naw.RandomWordAug(action='swap')
            # augmented_text3 = X.apply(aug_rws.augment)
            # df = pd.DataFrame()
            # df['text'] = augmented_text3
            # df['stars'] = y
            # df.to_csv(self.PATH + 'swap_aug.csv', index=False)

        elif self.aug_num == '3':

            print("Running first augmentation...")
            aug = nac.KeyboardAug()
            augmented_text = X.apply(aug.augment)
            df = pd.DataFrame()
            df['text'] = augmented_text
            df['stars'] = y
            df.to_csv(self.PATH + 'typo_aug.csv', index = False)

            print("Running second augmentation...")
            aug_del = naw.RandomWordAug(action='delete')
            augmented_text4 = X.apply(aug_del.augment)
            df = pd.DataFrame()
            df['text'] = augmented_text4
            df['stars'] = y
            df.to_csv(self.PATH+'delete_aug.csv', index=False)

        print("Done.")

if __name__ == "__main__":
    args = sys.argv
    aug_num = args[1]
    DataAugmentation(args[1]).run()


