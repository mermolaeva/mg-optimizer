
import re

from datetime import datetime, timedelta
from collections import UserList, namedtuple, Sequence, Counter, OrderedDict
from pprint import pprint, pformat
from math import log2
from itertools import product, chain, combinations

palette_hex = ["#F7931D", "#00B9F1", " #00A875", "#ECDE38", " #0072BC", "#F15A22", " #DA6FAB"]
default_hex = "#000000"

def require_input():
    input("Enter to continue...")
    
exception_ind = "An LI produced the same expression at different steps. This shouldn't be possible."

def pf(obj):
    if isinstance(obj, Feature):
        return "{}{}{}".format(obj[0][0], obj[1], obj[0][1])
    # elif isinstance(obj, Bundle):
    #     return " ".join([pf(x) for x in obj])
    else:
        return pformat(obj)
        
def pprint_tree(t, sep="", inherit=""): # pretty-print n-ary tree in preorder; assuming (node_label, is_terminal_below)
    parent = pf(t[0][0])
    if t[0][1] == True: # terminal node directly below
        print("{}╴{}  '{}'".format(sep, parent,pname(t[1])))
    else: # non-terminal node directly below
        print("{}╴{}".format(sep, parent))
        for c in t[1:-1]:
            pprint_tree(c, inherit+" ├──", inherit+" │  ")
        pprint_tree(t[-1], inherit+" └──", inherit+"    ")
                
def pprint_trie(d, sep="", inherit="", parent=""): # pretty-print trie
    if not isinstance(d, dict):
        print("{}╴{} :: {}".format(sep, pname(parent[0]), pf(parent[1])))
    else:
        print("{}╴{}".format(sep, pf(parent) if parent else "*"))
        for i, key in enumerate(d):
            if i == len(d)-1: pprint_trie(d[key], inherit+" └──", inherit+"    ", key)
            else: pprint_trie(d[key], inherit+" ├──", inherit+" │  ", key)

def pretty_yield(s, sep=" "):
    words = [" ".join([m[0] for m in hlist if not m[0] in silent]) for hlist in s]
    return sep.join([w for w in words if w])
    
def pretty_columns(data):
    if data:
        col_widths = [max(len(word) for word in col)+3 for col in zip(*data)]
        for row in data:
            print ("".join(word.ljust(col_widths[i]) for i, word in enumerate(row)))

def pretty_mg(mg, final=False):
    sorted_pretty = sorted(mg.items(), key=lambda x: ppretty(x[0]))
    max_show = 500
    pretty_columns([(pname(key), ppretty_final(key) if final else ppretty(key), "::", pf(val)) for key, val in sorted_pretty[:max_show]])
    if len(sorted_pretty) > max_show:
        print("... [{} more]".format(len(sorted_pretty)-max_show))
    print()
        
def pretty_eqs(eqs):
    for k, v in sorted(eqs.items(), key=lambda x: pphon(x[0])):
        print("{} = {}".format(" + ".join(pname(mu) for mu in v), pname(k)))
    print()
    
def pretty_cfg(cfg, split_lex=False, used=False, nonzero=False):
    if cfg != None:
        for left, rights in cfg.items():
                        
            right_lex = len([right for right in cfg[left].values() if right.is_term])
            right_der = len(rights)-right_lex
            if split_lex and (right_lex > 1) and (right_der > 0):
                cost_lex = "(log2({}) + log2({}))".format(right_der+1, right_lex)
                cost_der = "log2({})".format(right_der+1)
            else:
                cost_lex = cost_der = "log2({})".format(len(rights))
                
            for right in rights:
                if used==False or rights[right].usage_sum() > 0:
                    if nonzero==False or len(rights) > 1:
                        print("{} --> {}\t{}\t {} x {}".format(pf(left), " ".join(pf(r) for r in right), rights[right].usage, rights[right].usage_sum(), cost_lex if rights[right].is_term else cost_der))
        print()
            
def pretty_exp(exp):
    print(", ".join(" ".join(ppretty_str(w[1]) for w in ch) for ch in exp))
            
def pretty_sentence(s):
    print(" ".join(ppretty_str(m[1]) for m in s))

class Bundle(Sequence):
    def __init__(self, a):
        self.tup = tuple(a)
        
    def __repr__(self):
        return " ".join([pf(f) for f in self])

    def __len__(self):
        return len(self.tup)

    def __getitem__(self, index):
        return self.tup[index]
        
    def __eq__(self, other):
        return self.tup == other.tup
        
    def __gt__ (self, other):
        return other.tup < self.tup
        
    def __lt__ (self, other):
        return other.tup > self.tup
    
    def __hash__(self):
        return hash(repr(self)) # TODO: fix this!! Hash should have nothing to do with pretty-printing!
        
    def __add__(self, other):
        return self.tup + other
        
    def __radd__(self, other):
        return other + self.tup
        
        
Feature = namedtuple('Feature', 'type val')

rsel, lsel, ssel, wsel, cat = ("=", ""), ("", "="), ("=>", ""), ("<=", ""), ("", "")
lic, clic, lee  = ("+", ""), ("*", ""), ("-", "")
usel, ulic = ("?=", ""), ("?+", "")

types = [rsel, lsel, ssel, wsel, cat, lic, clic, lee]
type_chars = set("".join((t[0]+t[1]) for t in types))
symbol_init, symbol_prefix, tmp_suffix, cov_suffix  = "S", "t", "_tmp", "_cov"

atomic, derived, covert = "::", ":", "⋮"
atomic = derived
exp_types = [atomic, derived, covert]

is_sel = lambda x: x.type in [rsel, lsel, ssel, wsel]
is_msel = lambda x: x.type in [ssel, wsel]
is_nmsel = lambda x: x.type in [rsel, lsel]
is_cat = lambda x: x.type in [cat]
is_lic = lambda x: x.type in [lic, clic]
is_lee = lambda x: x.type in [lee]
is_u = lambda x: x.type in [usel, ulic]
is_affix = lambda b: is_msel(b[0]) # given a (non-empty) bundle, check if it is an affix

temp_batch = "B"
temp_nonbatch = "NB"
temp_b_i = lambda i: "{}{}".format(temp_batch, i)
temp_nb_i_j = lambda i: lambda j: "{}{}{}".format(temp_nonbatch, i, j)
is_temp_batch = lambda s: s.startswith(temp_batch)

emp = "ε"
emp_mor = "μ"
emp_true = ""
orig_int = lambda s: "F{}".format(s)
trace_hm = "" # head movement trace
trace_ph = "ε" # phrasal movement trace
silent = [emp, emp_mor, trace_hm, trace_ph]
start_symbol = "S"
transp_symbols = '_0123456789'+emp+emp_mor
mcfg_template = "{} --> {} {} {}" # left, right, mcfg map, comment
nstart, nend = "start", "end"

pclean = lambda x: "".join(c for c in x[0] if not c==emp) # TODO: simplify!
pphon = lambda x: "".join(c for c in x[0] if not c in silent)
ppretty = lambda x: x[0] if x[0] else emp
ppretty_final = lambda x: x[0] if x[0] and not x[0] in silent else emp
ppretty_str = lambda x: x if x and not x in silent else emp
pname = lambda x: "{}{}".format("" if x[0] else emp, "_".join(str(el) for el in x)) # add ε if necessary

def in_dict(d, key, val):
    if key in d and val in d[key]: return True
    else: return False

def dict_append(d, key, val, as_set=False):
    try:
        if as_set: d[key].add(val)
        elif not val in d[key]: d[key].append(val) # disallow repetitions
    except KeyError: d[key] = set([val]) if as_set else [val]
    
def dict_append_rep(d, key, val):
    try: d[key].append(val) # allow repetitions
    except KeyError: d[key] = [val]
    
def li_name(phon, d, extra=None):
    if not phon in d:
        new_name = (phon,"0") if phon in silent else (phon,)
        d[phon] = 1
    else:
        new_name = (phon, "{}".format(d[phon]))
        d[phon] += 1
    if extra != None: new_name = new_name + (extra,)
    return new_name, d

def common_index(strings, suffix=False):
    if suffix: strings = [s[::-1] for s in strings]
    s1, s2 = sorted((min(strings), max(strings)), key=len)
    ind = 0
    for i, c in enumerate(s1):
        if c == s2[i]:
            ind = -i-1 if suffix else i+1
        else: break
    return ind

def common_affix(strings, suffix=False):
    ind = common_index(strings, suffix)
    return get_shared(list(strings)[0], ind, suffix)
    
def get_slice(seq, i, rev, left):
    if left: return seq[:len(seq)+i] if rev else seq[:i]
    else: return seq[-1:i-1:-1][::-1] if rev else seq[i:]

lcp = lambda l: common_affix(l, False)
lcs = lambda l: common_affix(l, True)

get_shared = lambda seq, i, rev: get_slice(seq, i, rev, not rev)
get_remaining = lambda seq, i, rev: get_slice(seq, i, rev, rev)
get_slice_pos = lambda seq, i, left: seq[:i] if left else seq[i+1:] # positive indices only

get_cat = lambda b: next((x.val for x in b if is_cat(x)), None)
get_msel = lambda b: next((x.val for x in b if is_msel(x)), None)
get_first_sel = lambda b: b[0] if is_sel(b[0]) else Feature(None, None)

def attach_slice(seq, item, left):
    new_seq = list(seq)
    if left: new_seq.insert(0, item)
    else: new_seq.append(item)
    return new_seq

attach_shared = lambda seq, item, rev: attach_slice(seq, item, rev)
attach_remaining = lambda seq, item, rev: attach_slice(seq, item, not rev)
concat_shared = lambda seqs, rev: sum(seqs if rev else reversed(seqs), ())

to_pos = lambda i, s: i % len(s) if i < 0 else len(s) if i==inf else i # convert negative string indices to positive
neg_to_pos = lambda i, s: i % len(s) if i < 0 else len(s) if i==0 else i # convert negative string indices to positive

def powerset(l):
    return chain.from_iterable(combinations(l, r) for r in range(1, len(l)+1))
