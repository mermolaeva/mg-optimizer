
# MG to MCFG, file to MG, MG to file, sentence generators

from utils import *
from itertools import islice
from sys import maxsize
import random

title_line = '{} {{}}'.format(datetime.now().isoformat(' ', 'seconds')) # curr_name

template_mg = '''/{}/

/Start category/
{};

{}
''' # name, start category, LIs

class Chain: # type and syntactic feature configuration
    def __init__(self, type, features):
        self.type = type # atomic, derived, covert
        self.features = Bundle(features)            
    def __repr__(self):
        return "{} {}".format(self.type, pf(self.features))        
    def __members(self):
        return (self.type, self.features)
    def __eq__(self, other):
        return type(other) is type(self) and self.__members() == other.__members()
    def __hash__(self):
        return hash(self.__members())
    def __len__(self):
        return len(self.features)
        
start_exp_fun = lambda hn: (Chain(derived, Bundle([Feature(cat, hn),])),) # NOTE: assuming that :: s is not a thing
        
class Rule_right:
    def __init__(self, fst, snd=None):
        self.fst = fst # an expression or morpheme name
        self.snd = snd # an expression or None, if the rule is unary
        
    def __repr__(self):
        return str((self.fst, self.snd))    
        
class Rule_data:
    def __init__(self, is_term, usage, mcfg_map=None):
        self.is_term = is_term
        self.usage = usage
        self.mcfg_map = mcfg_map
        
    def __repr__(self):
        return str([self.is_term, self.mcfg_map, self.usage])
        
    def usage_sum(self):
        return sum(val.num for val in self.usage.values())
        
class LI_usage:
    def __init__(self, ind=None, num=0):
        self.ind = ind
        self.num = num
        
    def __repr__(self):
        return str((self.ind, self.num))

def map_movers(exp, num, ind=None, rep=False): # expression, first coordinate, index to exclude, empty replacement
    mover_range = range(3, len(exp)+2)
    if rep: result = tuple(((num, i),) if i != ind else None for i in mover_range)
    else: result = tuple(((num, i),) for i in mover_range if i != ind)
    return result

# MCFG maps
map_mrg_right = lambda m1, m2: (((0,0),), ((0,1),), ((0,2), (1,0), (1,1), (1,2))) + m1 + m2
map_mrg_right_s = lambda m1, m2: (((0,0),), ((1,1), (0,1)), ((0,2), (1,0), (1,2))) + m1 + m2
map_mrg_right_w = lambda m1, m2: (((0,0), (0,2), (1,0)), ((1,1), (0,1)), ((1,2),)) + m1 + m2
map_mrg_left = lambda m1, m2: (((1,0), (1,1), (1,2), (0,0)), ((0,1),), ((0,2),)) + m1 + m2
map_mrg_nonfinal = lambda m1, m2: (((0,0),), ((0,1),), ((0,2),)) + m1 + (((1,0), (1,1), (1,2)),) + m2
map_mrg_right_cov = lambda m1, m2: (((0,0),), ((0,1),), ((0,2), (1,0), (1,1), (1,2))) + m1 + (None,) + m2
map_mrg_left_cov = lambda m1, m2: (((1,0), (1,1), (1,2), (0,0)), ((0,1),), ((0,2),)) + m1 + (None,) + m2
map_mv_final = lambda m, i: (((0,i), (0,0)), ((0,1),), ((0,2),)) + m # m lacks i-th mover
map_mv_nonfinal = lambda m: (((0,0),), ((0,1),), ((0,2),)) + m # m has all movers intact
map_mv_cov = lambda m, i: (((0,i), (0,0)), ((0,1),), ((0,2),)) + m # m has None instead of i-th mover
map_start = (((0,0), (0,1), (0,2)),)

def mrg_right(exp1, exp2, is_mcfg=True): # final right merge
    newsubexp = Chain(derived, exp1[0].features[1:])
    mcfg_map = map_mrg_right(map_movers(exp1, 0), map_movers(exp2, 1)) if is_mcfg else None
    result = (newsubexp,) + exp1[1:] + exp2[1:]
    return result, mcfg_map

def mrg_right_s(exp1, exp2, is_mcfg=True): # final right strong merge
    newsubexp = Chain(derived, exp1[0].features[1:])
    mcfg_map = map_mrg_right_s(map_movers(exp1, 0), map_movers(exp2, 1)) if is_mcfg else None
    result = (newsubexp,) + exp1[1:] + exp2[1:]
    return result, mcfg_map

def mrg_right_w(exp1, exp2, is_mcfg=True): # final right weak merge
    newsubexp = Chain(derived, exp1[0].features[1:])
    mcfg_map = map_mrg_right_w(map_movers(exp1, 0), map_movers(exp2, 1)) if is_mcfg else None
    result = (newsubexp,) + exp1[1:] + exp2[1:]
    return result, mcfg_map

def mrg_left(exp1, exp2, is_mcfg=True): # final left merge
    newsubexp = Chain(derived, exp1[0].features[1:])
    mcfg_map = map_mrg_left(map_movers(exp1, 0), map_movers(exp2, 1)) if is_mcfg else None
    result = (newsubexp,) + exp1[1:] + exp2[1:]
    return result, mcfg_map

def mrg_nonfinal(exp1, exp2, is_mcfg=True): # nonfinal merge
    newsubexp = Chain(derived, exp1[0].features[1:])
    newmover = Chain(derived, exp2[0].features[1:])
    mcfg_map = map_mrg_nonfinal(map_movers(exp1, 0), map_movers(exp2, 1)) if is_mcfg else None
    result = (newsubexp,) + exp1[1:] + (newmover,) + exp2[1:]
    return result, mcfg_map
    
def mrg_right_cov(exp1, exp2, is_mcfg=True): # nonfinal right merge of a covert mover
    newsubexp = Chain(derived, exp1[0].features[1:])
    newmover = Chain(covert, exp2[0].features[1:])
    mcfg_map = map_mrg_right_cov(map_movers(exp1, 0), map_movers(exp2, 1)) if is_mcfg else None
    result = (newsubexp,) + exp1[1:] + (newmover,) + exp2[1:]
    return result, mcfg_map

def mrg_left_cov(exp1, exp2, is_mcfg=True): # nonfinal left merge of a covert mover
    newsubexp = Chain(derived, exp1[0].features[1:])
    newmover = Chain(covert, exp2[0].features[1:])
    mcfg_map = map_mrg_left_cov(map_movers(exp1, 0), map_movers(exp2, 1)) if is_mcfg else None
    result = (newsubexp,) + exp1[1:] + (newmover,) + exp2[1:]
    return result, mcfg_map

def mv_final(exp, ind, is_mcfg=True): # exp: expression, ind: mover's index; final move
    newsubexp = Chain(derived, exp[0].features[1:])
    mcfg_map = map_mv_final(map_movers(exp, 0, ind+2), ind+2) if is_mcfg else None
    result = (newsubexp,) + exp[1:ind] + exp[ind+1:]
    return result, mcfg_map
    
def mv_nonfinal(exp, ind, is_mcfg=True): # exp: expression, ind: mover's index; nonfinal move
    newsubexp = Chain(derived, exp[0].features[1:])
    newmover = Chain(derived, exp[ind].features[1:])
    mcfg_map = map_mv_nonfinal(map_movers(exp, 0)) if is_mcfg else None
    result = (newsubexp,) + exp[1:ind] + (newmover,) + exp[ind+1:]
    return result, mcfg_map
    
def mv_nonfinal_cov(exp, ind, is_mcfg=True): # exp: expression, ind: mover's index; nonfinal move of a covert mover
    newsubexp = Chain(derived, exp[0].features[1:])
    newmover = Chain(covert, exp[ind].features[1:])
    mcfg_map = map_mv_cov(map_movers(exp, 0, ind+2), ind+2) if is_mcfg else None
    result = (newsubexp,) + exp[1:ind] + (newmover,) + exp[ind+1:]
    return result, mcfg_map
    
def smc_compliant(exp):
    nchain_names = [c.features[0].val for c in exp[1:]] # first feature names of all non-initial chains
    return len(nchain_names) == len(set(nchain_names)) # check uniqueness
    
fun = { # {(positive feature type, whether selectee/licensee will move) : [function(s) to run]}
(rsel, True):[mrg_nonfinal, mrg_right_cov],
(rsel, False):[mrg_right],
(lsel, True):[mrg_nonfinal, mrg_left_cov],
(lsel, False):[mrg_left],
(ssel, True):None, # TODO: implement this, just in case
(ssel, False):[mrg_right_s],
(wsel, True):None, # TODO: implement this, just in case
(wsel, False):[mrg_right_w],
(lic, True):[mv_nonfinal, mv_nonfinal_cov],
(lic, False):[mv_final],
(clic, True):[mv_nonfinal, mv_nonfinal_cov],
(clic, False):[mv_final],
}

def cfg_to_exps(cfg):
    feat_to_exp = {}
    for exp in cfg.keys():
        if exp != start_symbol:
            dict_append(feat_to_exp, exp[0].features[0], exp, True)
    return feat_to_exp

def closure(mcfg_dict, feat_to_exp):
    
    while True: # close the set of expressions under Merge and Move
        new_exps = set()
        for first_feat, exps in feat_to_exp.items():
            
            if is_sel(first_feat): # apply Merge
                for exp in exps: # iterate over exps
                    try: matches = feat_to_exp[Feature(cat, first_feat.val)]
                    except KeyError: matches = []
                    for m in matches: # for every matching expression
                        will_move = len(m[0].features) > 1
                        for f in fun[(first_feat.type, will_move)]:
                            result, mcfg_map = f(exp, m)
                            new_exps.add(result)
                            mcfg_dict.setdefault(result, {}) # if the LHS is not present
                            mcfg_dict[result].setdefault((exp, m), Rule_data(False, {}, mcfg_map)) # if the RHS is not present

            elif is_lic(first_feat): # apply Move
                for exp in exps: # iterate over exps
                    move_indices = [ind for ind in range(1, len(exp)) if is_lee(exp[ind].features[0]) and exp[ind].features[0].val == first_feat.val]
                    if len(move_indices) == 1: # check for a unique, matching licensee
                        mover_ind = move_indices[0]
                        if (exp[mover_ind].type==covert) == (first_feat.type==clic): # make sure mover type matches licensor type
                            will_move_again = len(exp[mover_ind].features) > 1
                            for f in fun[(first_feat.type, will_move_again)]:
                                result, mcfg_map = f(exp, mover_ind)
                                new_exps.add(result)
                                mcfg_dict.setdefault(result, {}) # if the LHS is not present
                                mcfg_dict[result].setdefault((exp,), Rule_data(False, {}, mcfg_map)) # if the RHS is not present

        updated = False
        for new_exp in new_exps:
            new_first_feat = new_exp[0].features[0]
            if smc_compliant(new_exp) and not in_dict(feat_to_exp, new_first_feat, new_exp): # enforce SMC, check if exp is new
                dict_append(feat_to_exp, new_first_feat, new_exp, True)
                updated = True
        if not updated: break

    return mcfg_dict

def mg2mcfg(mg, start_exp):
    feat_to_exp = {}
    mcfg_dict = {}
    
    for right, left in mg.items(): # generate terminal rules and expressions
        li_exp = (Chain(atomic, left),)
        dict_append(feat_to_exp, left[0], li_exp, True)
        mcfg_dict.setdefault(li_exp, {})
        mcfg_dict[li_exp][(right,)] = Rule_data(True, {})
    
    mcfg_dict = closure(mcfg_dict, feat_to_exp) # generate nonterminal rules

    mcfg_dict[start_symbol] = {(start_exp,):Rule_data(False, {}, map_start)} # add start rule; assuming :: s does NOT exist
    mcfg_dict = rule_heads(mcfg_dict) # record possible head LIs
        
    return mcfg_dict

def rule_heads(mcfg): # record possible head LIs for each LHS expression; should not overwrite existing usage data
    rule_heads_aux.mcfg = mcfg
    rule_heads_aux.exp_to_heads = {}
    rule_heads_aux(start_symbol, set())
    return rule_heads_aux.mcfg
    
def rule_heads_aux(left, ancestors):
    left_heads = {} # all LIs that can be the head of this expression
    if left in ancestors: return
    ancestors.add(left)
    
    for right in rule_heads_aux.mcfg[left]: # rule is fully determined by left and right
        if right[0] in rule_heads_aux.exp_to_heads:
            heir_heads = rule_heads_aux.exp_to_heads[right[0]]
        elif rule_heads_aux.mcfg[left][right].is_term == True:
            heir_heads = {right[0]:-1}    
        else:
            heir_heads = rule_heads_aux(right[0], ancestors)
        allowed_heads = {m:i+1 for m, i in heir_heads.items()}
            
        for right_arg in right[1:]: # run the function for nonfirst argument(s)
            if right_arg not in rule_heads_aux.exp_to_heads:
                rule_heads_aux(right_arg, ancestors)

        for m, i in allowed_heads.items():
            left_m = left_heads.setdefault(m, i) # update list of heads for the left-hand side expression
            rule_m = rule_heads_aux.mcfg[left][right].usage.setdefault(m, LI_usage(i, 0)) # update usage of the rule (right, left)
            if not (left_m == rule_m.ind == i):
                raise Exception("An LI produced the same expression at different steps. This shouldn't be possible.")
    
    rule_heads_aux.exp_to_heads[left] = left_heads
    return left_heads
    
# TODO: rewrite the entire dang thing so it doesn't modify existing rules!
def rules_useful(rules, qreach_init=start_symbol): # remove useless rules, given a starting reachable symbol
    rules_reachable, rules_generating = {}, {}
    qreach = set(qreach_init)
    qgen = set()
    
    while qreach: # collect reachable rules
        new_qreach = set()
        for exp in qreach:
            if exp in rules:
                rules_reachable[exp] = rules[exp]
                for t in rules[exp]:
                    if rules[exp][t].is_term == True: qgen.update(t) # terminal rule
                    else: new_qreach.update(t)
                del rules[exp]
        qreach = new_qreach

    new_len = len(qgen)
    while True: # collect generating rules
        for exp in rules_reachable:
            to_delete = set()
            for t in rules_reachable[exp]:
                if qgen.issuperset(t):
                    if not exp in rules_generating: rules_generating[exp] = {}           
                    rules_generating[exp][t] = rules_reachable[exp][t]
                    qgen.add(exp)
                    to_delete.add(t)
            for t in to_delete:
                del rules_reachable[exp][t]        
        curr_len, new_len = new_len, len(qgen)
        if new_len == curr_len: break
    
    return rules_generating
    
def gen_ord_syn(mcfg, mor_to_str, n):
    examples = gen_ord(mcfg, mor_to_str, n, expand_terms=False)
    examples = [add_fringe(x, mcfg, mor_to_str) for x in examples]
    return examples
    
def gen_sample_syn(mcfg, mor_to_str, n):
    examples = gen_sample(mcfg, mor_to_str, n, expand_terms=False)
    examples = [add_fringe(x, mcfg, mor_to_str) for x in examples]
    return examples

def gen_rand(mcfg, mor_to_str, n):
    return [random_tree(mcfg, mor_to_str) for i in range(n)]

def gen_ord(mcfg, mor_to_str, n, expand_terms=True):
    iter = gen_ord_all(mcfg, mor_to_str, expand_terms)
    examples = []
    for x in iter:
        examples.append(x[0])
        if len(examples) == n: break
    return examples
    
def gen_sample(mcfg, mor_to_str, n, expand_terms=True):
    pop = [x[0] for x in gen_ord_all(mcfg, mor_to_str, expand_terms)]
    return random.sample(pop, min(n, len(pop)))

def weighted_choice(weights, t, factor=10):
    rights_zip = list(zip(*weights[t].items()))
    children = random.choices(rights_zip[0], weights=rights_zip[1], k=1)[0]
    for r in weights[t]:
        if r != children: weights[t][r] += factor # increase weights for each alternative rule t --> ...
    return weights, children
    
def random_tree(mcfg, mor_to_str, weighted=True): # produce a random derivation tree given an (M)CFG
    expand_nonterm.mcfg = mcfg
    expand_nonterm.weighted = weighted
    expand_nonterm.mor_to_str = mor_to_str
    if weighted:
        expand_nonterm.weights = {l:{r:1 for r, val in mcfg[l].items()} for l in mcfg}
    result = expand_nonterm(start_symbol)
    return result

def expand_nonterm(t): # randomly expand nonterminal t
    if expand_nonterm.weighted:
        expand_nonterm.weights, children = weighted_choice(expand_nonterm.weights, t)
    else:
        children = random.choice(tuple(expand_nonterm.mcfg[t]))
    is_term = expand_nonterm.mcfg[t][children].is_term
    result = [(t, is_term)]
    for c in children:
        if is_term: result.append((c, expand_nonterm.mor_to_str[c]))
        else: result.append(expand_nonterm(c))
    return result
    
def add_fringe(t, mcfg, mor_to_str):
    add_fringe_aux.mcfg = mcfg
    add_fringe_aux.mor_to_str = mor_to_str
    add_fringe_aux.weights = {l:{r:1 for r, val in mcfg[l].items() if val.is_term} for l in mcfg}
    t = add_fringe_aux(t)
    return t

def add_fringe_aux(t): # traverse t; when reaching a leaf, pick one using weights
    new_t = [t[0],]
    children = tuple(t[1:])
    if t[0][1] == True: # only child is a leaf
        # add_fringe_aux.weights, children = weighted_choice(add_fringe_aux.weights, t[0][0])
        # print(add_fringe_aux.mcfg[t[0][0]])
        children = random.choice([r for r, val in add_fringe_aux.mcfg[t[0][0]].items() if val.is_term])
        # t[1] = (children[0], add_fringe_aux.mor_to_str[children[0]])
        new_t.append((children[0], add_fringe_aux.mor_to_str[children[0]]))
    else:
        # for c in children:
        #     add_fringe_aux(c)
        new_t.extend([add_fringe_aux(c) for c in children])
    return new_t
    
def gen_ord_all(mcfg, mor_to_str, expand_terms): # produce all trees in order
    generate_all.mcfg = mcfg    
    generate_all.mor_to_str = mor_to_str
    use_hash_start = {lhs:0 for lhs in mcfg}
    # use_hash_start = {(lhs, rhs):0 for lhs, val in mcfg.items() for rhs in val}    
    # return [x[0] for x in generate_all([start_symbol,], maxsize, use_hash_start)]
    return generate_all([start_symbol,], maxsize, use_hash_start, expand_terms)

def generate_all(items, depth, use_hash, expand_terms):
    if items: # there are symbols left to be expanded
        for frag1 in generate_one(items[0], depth, use_hash, expand_terms): # all options for expanding first symbol in the list
            for frag2 in generate_all(items[1:], depth, use_hash, expand_terms): # all options for expanding remaining symbols
                yield [frag1,] + frag2
    else: yield []

def generate_one(item, depth, use_hash_parent, expand_terms):
    if depth > 0:

        if (not expand_terms) and any(generate_all.mcfg[item][rhs].is_term for rhs in generate_all.mcfg[item]):
            yield [(item, True), ((), ())]
        
        for rhs in generate_all.mcfg[item]:
            use_hash = {key:val for key, val in use_hash_parent.items()} # copy the dictionary
            
            if use_hash[item] < 2: # how many times the same expression is allowed on the path from the root
                use_hash[item] += 1
            # if use_hash[(item, rhs)] < 1: # how many times the same rule is allowed on the path from the root
            #     use_hash[(item, rhs)] += 1
                if generate_all.mcfg[item][rhs].is_term:
                    if expand_terms: yield[(item, True), (rhs[0],  generate_all.mor_to_str[rhs[0]])]
                    else: pass
                else:
                    for frag in generate_all(rhs, depth - 1, use_hash, expand_terms):
                        yield [(item, False),] + frag

def mcfg_uses(t, mcfg): # count rule uses per head LI, updating an (M)CFG
    mcfg_uses_aux.mcfg = mcfg
    mcfg_uses_aux(t)
    return mcfg_uses_aux.mcfg
        
def mcfg_uses_aux(t):
    children = tuple(t[1:])
    
    if t[0][1] == True: # only child is a leaf
        t_usage = mcfg_uses_aux.mcfg[t[0][0]][(t[1][0],)].usage
        head_li = t[1][0]
        head_ind = 0
    else:
        (child_head_lis, child_head_inds) = zip(*[mcfg_uses_aux(c) for c in children])
        child_names = tuple(c[0][0] for c in children)
        t_usage = mcfg_uses_aux.mcfg[t[0][0]][child_names].usage
        head_li = child_head_lis[0]
        head_ind = child_head_inds[0] + 1
        
    if (not head_li in t_usage):
        raise Exception("This expression: {} cannot be headed by this LI: {}".format(t[0][0], head_li))
    elif t_usage[head_li].ind != head_ind:
        raise Exception(exception_ind)
    else: t_usage[head_li].num += 1
    
    return head_li, head_ind

def mcfg_string(t, mcfg): # produce a derived tuple of strings from derivation tree, given an MCFG
    mcfg_string_aux.mcfg = mcfg
    return mcfg_string_aux(t)[0]
        
def mcfg_string_aux(t):
    children = tuple(t[1:])
    if t[0][1] == True: # only child is a leaf
        t_exp = (tuple(), (children[0],), tuple())
    else:
        child_names = tuple(c[0][0] for c in children)
        child_exps = [mcfg_string_aux(c) for c in children]
        t_map = mcfg_string_aux.mcfg[t[0][0]][child_names].mcfg_map
        t_exp = mcfg_concat(t_map, child_exps)
    return t_exp
    
def mcfg_concat(rule_map, exp_list): # combine a list of expressions according to the MCFG map
    exp = []
    for component in rule_map:
        if component != None:
            exp.append(tuple(x for pair in component for x in exp_list[pair[0]][pair[1]]))
        else: exp.append(tuple())
    return tuple(exp)
    
def mcfg_to_words(s, mg):
    l, curr = [], []
    for m in s:
        if is_affix(mg[m[0]]):
            curr.append(m[0])
        else:
            if curr: l.append(tuple(curr))
            curr = [m[0],]
    if curr: l.append(tuple(curr))
    return l
    
def mcfg_to_pf(s, mg):
    l = ["{}{}".format("-" if is_affix(mg[m[0]]) else "", ppretty_str(m[1])) for m in s]
    return " ".join(l)

def mcfg_to_pf_simple(s):
    return " ".join(ppretty_str(m[1]) for m in s)

def cfg_check(cfg, eqs, verbose=False):
    # cfg = rules_useful(cfg) # TODO: rewrite and use
    cfg_check_aux.term, cfg_check_aux.nonterm, cfg_check_aux.selectees = {}, {}, {}
    cfg_check_aux.verbose=verbose
    if cfg_check_aux.verbose: print("Verbose mode")
    
    term_usage = {}
    for word, morphemes in eqs.items():
        for m in morphemes:
            term_usage[m] = term_usage.setdefault(m, 0) + word[2]

    for l in cfg:
        for r, val in cfg[l].items():
            if val.is_term:
                if val.usage_sum() != term_usage[r[0]]:
                    if verbose:
                        print("Terminal usage problem: {} -- actual {} vs. expected {}".format(val, val.usage_sum(), term_usage[r[0]]))
                    return False
                else: cfg_check_aux.term[(l, r)] = val.usage_sum()
            else:
                if l != start_symbol and val.usage != {}:
                    cfg_check_aux.nonterm[(l, r)] = dict(val.usage)
                if len(r) == 2 or l==start_symbol:
                    cfg_check_aux.selectees[r[-1]] = cfg_check_aux.selectees.get(r[-1], 0) + val.usage_sum()
    
    for ((b, exp), val_sum) in cfg_check_aux.term.items():
        if cfg_check_aux.verbose: print("Processing LI: {} {} {}".format(pf(b), pf(exp), val_sum))
        if not cfg_check_aux(b, exp[0], val_sum): return False
    
    if cfg_check_aux.verbose:
        print(cfg_check_aux.nonterm.items())
        print(cfg_check_aux.selectees.items())
    
    if any(v != {} for v in cfg_check_aux.nonterm.values()) or any(v != 0 for v in cfg_check_aux.selectees.values()):
        return False
    
    return True
    
def cfg_check_aux(exp, li, n):
    
    if cfg_check_aux.verbose: print("Expression: {} {} {}".format(pf(exp), pf(li), n))
    
    if exp[0].features[0].type == cat: # lhs will serve as second argument
        cfg_check_aux.selectees[exp] -= n

    else: # lhs will serve as first argument
        next_rules = [(lhs, cfg_check_aux.nonterm[(lhs, rhs)].pop(li).num) for (lhs, rhs) in cfg_check_aux.nonterm if rhs[0] == exp]
        if cfg_check_aux.verbose: print(next_rules)
            
        if (
            next_rules == [] or 
            sum ([x[1] for x in next_rules]) != n or 
            any(cfg_check_aux(lhs, li, n_share)==False for (lhs, n_share) in next_rules)
        ):
            return False

    return True

# structure of cfg: {left: {right: Rule_data(is_term, {LI: LI_usage(ind, num)}, None)}}
def drop_maps(mcfg): # get rid of MCFG maps
    return {l: {r: Rule_data(val.is_term, val.usage) for r, val in mcfg[l].items()} for l in mcfg}
    
def make_corpus(mg, eqs, mor_to_str, corpus_size, gen_method, start_exp):
    eqs_reverse = {tuple(morphemes):list(word) for word, morphemes in eqs.items()}    
    
    mcfg = mg2mcfg(mg, start_exp)
    examples = gen_method(mcfg, mor_to_str, corpus_size) # n smallest trees  
    
    for ex in examples:
        mcfg = mcfg_uses(ex, mcfg)
        # pprint_tree(ex)
        for w in mcfg_to_words(mcfg_string(ex, mcfg), mg):
            eqs_reverse[w][2] +=1    
    eqs_usage = {tuple(word):list(morphemes) for morphemes, word in eqs_reverse.items()}

    return mcfg, examples, eqs_usage

from copy import deepcopy
from mdl import mdl_corpus

def make_corpus_each(mg, eqs, mor_to_str, corpus_size, gen_method, start_exp):  
    
    mcfg = mg2mcfg(mg, start_exp)
    examples = gen_method(mcfg, mor_to_str, corpus_size) # n smallest trees  
    
    for ex in examples:
        mcfg_ex = deepcopy(mcfg)
        mcfg_ex = mcfg_uses(ex, mcfg_ex)
        pretty_sentence(mcfg_string(ex, mcfg_ex))
        pprint_tree(ex)
        print("Example cost: {}".format(mdl_corpus(mcfg_ex, False)))
        pretty_cfg(rules_useful(mcfg_ex), nonzero=True, used=True)

    return
    
def file_to_mg(curr_name): # read grammar from a .mg file
    file = open("lexica/{}.mg".format(curr_name), "r")
    start = None
    mg, reps = {}, {}
    for line in file:
        line = line.split("/", 1)[0].strip() # remove whitespaces and comments
        if line: # ignoring empty lines
            if start == None:
                l = line.strip(";")
                if len(l) == 1: start = l # NOTE: assumes that the start category is a single symbol
                else: raise Exception("Cannot identify the start category")
            else:
                for item in line.split(";"): # for each lexical item
                    if item:
                        (item_mor, item_syn) = item.split(":: ")
                        pieces = [x for x in item_syn.split(" ")]
                        features = []
                        item_mor, reps = li_name(item_mor.strip(" "), reps)
                        for piece in pieces: # parse each feature into type and name
                            first_i = next(i for i, x in enumerate(piece) if not x in type_chars)
                            last_i = next((i for i, x in enumerate(piece[first_i:]) if x in type_chars), len(piece))
                            name = piece[first_i:last_i]
                            if name.isdigit(): name = orig_int(name) # if the name is all digits: add a letter
                            type = (piece[:first_i], piece[last_i:])
                            features.append(Feature(type, name))
                        mg[item_mor] = Bundle(features)
    return mg, start
    
def mg_to_file(mg, curr_name, head_name): # write grammar into an .mg file
    lines = []
    for key, val in mg.items():
        feats = " ".join(pf(f) for f in val)
        lines.append("{}{}:: {};".format(pphon(key), " " if pphon(key) else "", feats))
    result = template_mg.format(title_line.format(curr_name), head_name, "\n".join(lines))
    f = open("lexica/{}.mg".format(curr_name), "w")
    f.writelines(result)
    f.close()