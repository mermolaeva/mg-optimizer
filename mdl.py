from math import log2, inf
from utils import pphon

mdl_full = lambda f1, f2: lambda mg, cfg: (f1(mg), f2(cfg))

hsize_grammar = lambda x: x[0] # grammar only; not calculating corpus cost
hsize_grammar_alt = lambda x: x[0] # grammar only; calculating corpus cost
hsize_corpus = lambda x: x[1] # corpus only
hsize_ord = lambda x: (x[1], x[0]) # corpus, then grammar
hsize_sum = lambda x: sum(x) # sum of corpus and grammar

get_feature_names = lambda mg: set(f.val for b in mg.values() for f in b)
get_feature_number = lambda mg: len(get_feature_names(mg))
# len_types = len(types)
len_types = 7 # right, left, and HM selectors; categories; overt and covert licensors; licensees
get_symbol_cost = lambda mg: log2(26 + len_types + get_feature_number(mg) + 1)

# naive grammar encoding: feature count
def count_naive(mg):
    lexicon_cost = 0
    for left, right in mg.items():
        lexicon_cost += (len(pphon(left)) + (2 * len(right)) + 1) # total number of all features, pronounced chars, and separators
    return lexicon_cost

# same-length encoding for each symbol in Sigma, Types, and Base, plus LI delimiter
def mdl_1d(mg):
    symbol_cost = get_symbol_cost(mg)
    lexicon_cost = 0
    sum_phon, sum_syn = 0, 0

    for left, right in mg.items(): # for each LI
        # add 
        lexicon_cost += symbol_cost * (len(pphon(left)) + (2 * len(right)) + 1)
        sum_phon += len(pphon(left))
        sum_syn += len(right)
    
    # print("Symbol cost: {}".format(symbol_cost))
    # print("Sum phon: {}".format(sum_phon))
    # print("Sum syn: {}".format(sum_syn))
    return lexicon_cost

# 2-dimensional grammar encoding: group strings by feature bundle; encourage reused bundles
def mdl_2d(mg):
    symbol_cost = get_symbol_cost(mg)
    lexicon_cost = 0
    
    mg_by_bundle = {}    
    for left, right in mg.items():
        mg_by_bundle.setdefault(right, [])
        mg_by_bundle[right].append(len(pphon(left)))
    
    for b, strs in mg_by_bundle.items(): # (bundle, list of lengths of associated strings)
        b_cost = symbol_cost * (2 * len(b) + 1) # pay for bundle + one separator
        str_cost = symbol_cost * (sum(strs) + len(strs)) # for each string, pay its length + separator
        lexicon_cost += (b_cost + str_cost)
    
    return lexicon_cost
    
# 3-dimensional grammar encoding; group strings by types then names of features in bundles; encourage reused templates of feature bundles
def mdl_3d(mg):
    symbol_cost = get_symbol_cost(mg)
    lexicon_cost = 0
    
    mg_by_types = {}
    for left, right in mg.items():
        b_types = tuple(f.type for f in right)
        b_vals = tuple(f.val for f in right)
        mg_by_types.setdefault(b_types, {})
        mg_by_types[b_types].setdefault(b_vals, [])
        mg_by_types[b_types][b_vals].append(len(pphon(left)))
        # if not b_types in mg_by_types: mg_by_types[b_types] = {}
        # dict_append_rep(mg_by_types[b_types], b_vals, len(pphon(left)))
    
    for b_types in mg_by_types: # (bundle, list of length of associated strings)
        b_types_cost = symbol_cost * (len(b_types) + 1) # pay for types + one separator
        lexicon_cost += b_types_cost
        
        for b_vals, strs in mg_by_types[b_types].items():
            b_vals_cost = symbol_cost * (len(b_vals) + 1) # pay for values + one separator
            str_cost = symbol_cost * (sum(strs) + len(strs)) # for each string, pay its length + separator
            lexicon_cost += (b_vals_cost + str_cost)
    
    return lexicon_cost

mdl_cfg = lambda cfg: mdl_corpus(cfg, False)
mdl_cfg_split = lambda cfg: mdl_corpus(cfg, True)

def mdl_corpus(cfg, split_lex):
    if cfg == None: return 0
    corpus_cost = 0

    for left in cfg:

        right_lex = [right_data.usage_sum() for right_data in cfg[left].values() if right_data.is_term]
        
        if split_lex and len(right_lex) > 1:
            left_cost = log2(len(cfg[left])-len(right_lex)+1)
            corpus_cost += sum(right_data.usage_sum() for right_data in cfg[left].values()) * left_cost
            corpus_cost += sum(right_lex) * log2(len(right_lex)) # pay for lexical usages
            
        else:
            left_cost = log2(len(cfg[left]))
            corpus_cost += sum(right_data.usage_sum() for right_data in cfg[left].values()) * left_cost
                    
    return corpus_cost