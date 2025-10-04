'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

def freq_href_feature(text, freq):
    return float(freq['href'])

def freq_congratulations_feature(text, freq):
    return float(freq['congratulations'])

def freq_won_feature(text, freq):
    return float(freq['won'])

def freq_exclusive_feature(text, freq):
    return float(freq['exclusive'])

def freq_free_feature(text, freq):
    return float(freq['free'])

def freq_now_feature(text, freq):
    return float(freq['now'])

def freq_rich_feature(text, freq):
    return float(freq['rich'])

def freq_fast_feature(text, freq):
    return float(freq['fast'])

def freq_quick_feature(text, freq):
    return float(freq['quick'])

def freq_win_feature(text, freq):
    return float(freq['win'])

def freq_prize_feature(text, freq):
    return float(freq['prize'])

def freq_bonus_feature(text, freq):
    return float(freq['bonus'])

def freq_lottery_feature(text, freq):
    return float(freq['lottery'])

def freq_cash_feature(text, freq):
    return float(freq['cash'])

def freq_urgent_feature(text, freq):
    return float(freq['urgent'])

def freq_credit_feature(text, freq):
    return float(freq['credit'])

def freq_password_feature(text, freq):
    return float(freq['password'])

def freq_deal_feature(text, freq):
    return float(freq['deal'])

def freq_gift_feature(text, freq):
    return float(freq['gift'])

def freq_exclusive_feature(text, freq):
    return float(freq['exclusive'])

def freq_discount_feature(text, freq):
    return float(freq['discount'])

def freq_offer_feature(text, freq):
    return float(freq['offer'])

def freq_winner_feature(text, freq):
    return float(freq['winner'])

def freq_promotion_feature(text, freq):
    return float(freq['promotion'])

def freq_refund_feature(text, freq):
    return float(freq['refund'])

def freq_mortgage_feature(text, freq):
    return float(freq['mortgage'])

def freq_investment_feature(text, freq):
    return float(freq['investment'])

def freq_unsubscribe_feature(text, freq):
    return float(freq['unsubscribe'])

def freq_secret_feature(text, freq):
    return float(freq['secret'])

def freq_trial_feature(text, freq):
    return float(freq['trial'])

def freq_hurry_feature(text, freq):
    return float(freq['hurry'])

def freq_payout_feature(text, freq):
    return float(freq['payout'])

def freq_prize_feature(text, freq):
    return float(freq['prize'])

def freq_claim_feature(text, freq):
    return float(freq['claim'])

def freq_www_feature(text, freq):
    return float(freq['www'])

def freq_china_feature(text, freq):
    return float(freq['china'])

def freq_export_feature(text, freq):
    return float(freq['export'])

def freq_popular_feature(text, freq):
    return float(freq['popular'])

def freq_forwarded_feature(text, freq):
    return float(freq['forwarded'])

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    feature.append(freq_href_feature(text, freq))
    feature.append(freq_congratulations_feature(text, freq))
    feature.append(freq_won_feature(text, freq))
    feature.append(freq_exclusive_feature(text, freq))
    feature.append(freq_free_feature(text, freq))
    feature.append(freq_now_feature(text, freq))
    feature.append(freq_rich_feature(text, freq))
    feature.append(freq_fast_feature(text, freq))
    feature.append(freq_quick_feature(text, freq))
    feature.append(freq_win_feature(text, freq))
    feature.append(freq_prize_feature(text, freq))
    feature.append(freq_bonus_feature(text, freq))
    feature.append(freq_lottery_feature(text, freq))
    feature.append(freq_cash_feature(text, freq))
    feature.append(freq_urgent_feature(text, freq))
    feature.append(freq_credit_feature(text, freq))
    feature.append(freq_password_feature(text, freq))
    feature.append(freq_deal_feature(text, freq))
    feature.append(freq_gift_feature(text, freq))
    feature.append(freq_exclusive_feature(text, freq))
    feature.append(freq_discount_feature(text, freq))
    feature.append(freq_offer_feature(text, freq))
    feature.append(freq_winner_feature(text, freq))
    feature.append(freq_promotion_feature(text, freq))
    feature.append(freq_refund_feature(text, freq))
    feature.append(freq_mortgage_feature(text, freq))
    feature.append(freq_investment_feature(text, freq))
    feature.append(freq_unsubscribe_feature(text, freq))
    feature.append(freq_secret_feature(text, freq))
    feature.append(freq_trial_feature(text, freq))
    feature.append(freq_hurry_feature(text, freq))
    feature.append(freq_payout_feature(text, freq))
    feature.append(freq_prize_feature(text, freq))
    feature.append(freq_claim_feature(text, freq))
    feature.append(freq_www_feature(text, freq))
    feature.append(freq_china_feature(text, freq))
    feature.append(freq_export_feature(text, freq))
    feature.append(freq_forwarded_feature(text, freq))

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)
