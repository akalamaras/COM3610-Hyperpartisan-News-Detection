import pandas as pd
from pandas import read_xml
import re

# https://github.com/chialun-yeh/SemEval2019/blob/4cf5b57960100a41943cbba60d7413b0bab100fd/utils.py
def clean_quotations(string):
	string = re.sub(r'[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", string)
	string = re.sub(r'[„“”]|(\'\')|(,,)', '"', string)
	return string


def clean_text(main_text):
	# Remove any URLs from the XML
	main_text = re.sub(r'(http\S+)|(www\S+)|(href)', ' ', main_text)
	# Remove Twitter references/ads etc. https://github.com/chialun-yeh/SemEval2019/blob/4cf5b57960100a41943cbba60d7413b0bab100fd/utils.py
	main_text = re.sub(r'Getty [Ii]mages?|Getty|[Ff]ollow us on [Tt]witter|MORE:|ADVERTISEMENT|VIDEO', ' ', main_text)
	# Remove @, #, etc.
	main_text = re.sub(r'@\S+|C\S+|\.{2,}', ' ', main_text)
	# remove anything within {} or [] or ().
	main_text = re.sub(r'\{[^}]*\}|\[[^]]*\]|\([^)]*\)', ' ', main_text)
	# TODO More
	return main_text
