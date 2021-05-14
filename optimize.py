
import argparse
import os, shutil
import shelve

import networkx as nx
import pydot as pd

from grammars import *
from mdl import *

class Step:
    def __init__(self, mg, order, eqs, solution, cfg, fresh, parent=None, rank=None, level=None, note=None):
        self.mg = mg
        self.order = order
        self.eqs = eqs
        self.solution = solution
        self.cfg = cfg
        self.fresh = fresh
        self.parent = parent
        self.rank = rank
        self.level = level
        self.note = note
        
    def __eq__(self, other):
        return self.rank == other.rank
            
    def __gt__ (self, other):
        return other.rank < self.rank

def mg_to_plot(mg, n, t="", fresh=set()): # grammar, path sans extension, title, fresh categories
    graph = pd.Dot(graph_type='digraph', label=t, fontsize='32', rankdir='RL', layout='dot', esep=5)
    graph.add_node(pd.Node(head_name, style='filled', fillcolor='lightgray',  fontsize='32')) # fillcolor='lightskyblue'
    counter = 0
    
    colors = {f:palette_hex[i%len(palette_hex)] for i, f in enumerate(fresh)}    
    edges = {}
    for li in mg.items():
        edges.setdefault(li[1], []).append(li[0])
        
    for (b, phons) in edges.items():
        li_cat = get_cat(b)
        li_first = get_first_sel(b).val
        
        node_cat = pd.Node(li_cat, fontsize='32')
        if li_first != None:
            node_first = pd.Node(li_first, fontsize='32')
        else:
            node_first = pd.Node("z{}".format(counter), label="", style='dashed', fontsize='32')
            counter += 1
        
        # if len(b) == 2 and is_msel(b[0]) and all(not pphon(x) for x in phons):
        #     label = "" # suppress label if only a category changer
        # else: label="{}\n{}".format(" | ".join([pname(x) for x in phons]), pf(b))
        label="{}\n{}".format(" |\n ".join([pname(x) for x in phons]), pf(b))
        edge_color = colors[li_cat] if li_cat in fresh else colors[li_first] if li_first in fresh else default_hex
        edge_width = 2.0 if edge_color != default_hex else 1.0
        graph.add_node(node_first)
        graph.add_node(node_cat)
        graph.add_edge(pd.Edge(node_first, node_cat, label=label,  fontsize='32', color=edge_color, penwidth=edge_width))
        
    graph.write_svg('{}.svg'.format(n))
    # graph.write_dot('{}.dot'.format(n))
    
def word_graph(mg):
    mg_adj, edge_labels = {}, {}
    dict_append(mg_adj, head_name, nend, True) # sentence category is always allowed to terminate words
    
    for li_mor, li_syn in mg.items():        
        li_cat_val = get_cat(li_syn)
        li_sels = [f for f in li_syn if is_sel(f)]
        
        if li_sels and is_msel(li_sels[0]): rem_sels, vstart = li_sels[1:], li_sels[0].val # affix; edge from first sel to cat
        else: rem_sels, vstart = li_sels[0:], nstart # not affix; edge from start to cat
        
        dict_append(mg_adj, vstart, li_cat_val, True)
        dict_append(edge_labels, (vstart, li_cat_val), li_mor, True)
        
        for li_sel in rem_sels: # all selectors except strong/weak if present
            dict_append(mg_adj, li_sel.val, nend, True) # end words from all selected categories
     
    return nx.DiGraph(mg_adj), edge_labels

def preprocess_mg(mg): # initialize mg, eqs, solution; identify existing complex eqs if any
    new_mg, eqs, mor_to_str = {}, {}, {}
    mcounter, wcounter = {}, {}
    
    for left, right in mg.items():
        new_name, mcounter = li_name(emp_mor, mcounter)  
        new_mg[new_name] = right
        mor_to_str[new_name] = pphon(left)

    G, edge_labels = word_graph(new_mg)

    for path in nx.all_simple_paths(G, source=nstart, target=nend):
        for word in product(*[edge_labels[(path[i], path[i+1])] for i in range (0, len(path)-2)]):
            word_phon = "".join(mor_to_str[m] for m in word)
            wcounter.setdefault(word_phon, 0)
            word_name, wcounter = li_name(word_phon, wcounter, extra=0)
            eqs[word_name] = word
                      
    return new_mg, eqs, mor_to_str
    
def step_to_mg(step_mg, mor_to_phon):
    mg, reps = {}, {}
    for li_mor, li_syn in step_mg.items():
        new_name, reps = li_name(mor_to_phon[li_mor], reps)
        mg[new_name] = li_syn
    return mg

def current_hash(x):
    return hash(x)

def hash_step(mg, eqs={}):
    tuple_mg = tuple(mg.items())
    tuple_eqs = tuple((w, tuple(m)) for w, m in eqs.items())
    return current_hash((tuple_mg, tuple_eqs))
    
def rename_cfg(cfg, feature_dict, li_dict): # rename features and LIs within a CFG
    if cfg == None: return
    
    new_cfg = {}
    
    for left in cfg:
        if left == start_symbol: new_left = left
        else: new_left = tuple(Chain(c.type, unify_bundle(c.features, feature_dict)) for c in left)
        
        new_cfg.setdefault(new_left, {})
        
        for right in cfg[left]:
            if cfg[left][right].is_term == True: # terminal node: rename LI
                new_right = (li_dict[right[0]],)
            else: # nonterminal node: rename features in expressions
                new_right = tuple(tuple(Chain(c.type, unify_bundle(c.features, feature_dict)) for c in r) for r in right)
            
            new_cfg[new_left].setdefault(new_right, Rule_data(cfg[left][right].is_term, {}, None))
            
            for li, data in cfg[left][right].usage.items():
                new_cfg[new_left][new_right].usage.setdefault(li_dict[li], LI_usage(data.ind, 0))
                new_cfg[new_left][new_right].usage[li_dict[li]].num += data.num

    return new_cfg
    
def make_step(mg, ord, eqs, mor_to_phon, cfg):
    new_mg, new_ord, new_mor_to_phon = {}, [], {}
    fnames, mnames, li_pairs, li_reps = {}, {}, {}, {}

    # Rename features and LIs, dropping full duplicates
    for old_name in ord: # renaming morphemes in original order
        li_phon = mor_to_phon[old_name] 
        li_syn = Bundle([Feature(f.type, f.val if f.val in orig_names else fnames.setdefault(f.val, str(len(fnames)))) for f in mg[old_name]])
        li_pair = (li_phon, li_syn)
        
        if not li_pair in li_pairs: # there is no LI representing li_pair; this one becomes the representative
            new_name, li_reps = li_name(pclean(old_name), li_reps)
            new_mg[new_name] = li_syn # update grammar
            new_ord.append(new_name) # update order
            li_pairs[li_pair] = new_name # record the representative
            new_mor_to_phon[new_name] = li_phon
        
        mnames[old_name] = li_pairs[li_pair] # use the representative to name the morpheme in equations    
    
    # Rename morphemes within equations
    for word, mor in eqs.items():
        eqs[word] = list(mnames[m] for m in mor) # rename morpheme sequence in place for sorting purposes
    
    # Form new equations, sorting by new morpheme sequence and retaining equation names
    new_eqs = {word:mor for word, mor in sorted(eqs.items(), key=lambda x: x[1])}
    
    # Rename features and LIs in the CFG
    new_cfg = rename_cfg(cfg, fnames, mnames) if cfg else None
    fresh = set(fnames[f] for f in fnames if is_temp_batch(f))
    
    return Step(new_mg, new_ord, new_eqs, new_mor_to_phon, new_cfg, fresh)
    
uni_types_sets = {tuple(set((wsel, ssel))):wsel}

def unify_type(l):
    s = tuple(set(l))
    if len(s) == 1: return s[0]
    try: return uni_types_sets[s]
    except KeyError: return None
    
def unify_name(names, id=None):
    name_set = set(names)
    if len(name_set) == 1: return next(iter(names))
    elif head_name in name_set: return head_name
    else:
        name_orig = name_set.intersection(orig_names)
        if len(name_orig) == 1: return next(iter(name_orig))
        elif id != None: return id
        else: return names[0]
        
def unify_shared(bundles, syn_i, is_high): 
    new_b = []
    rng = range (-1, syn_i-1, -1) if is_high else range(0, syn_i)
    for j in rng:
        jtype = unify_type([b[j].type for b in bundles])
        jname = unify_name([b[j].val for b in bundles], "0{}".format(j))
        jfeature = Feature(jtype, jname)
        if is_high: new_b.insert(0, jfeature)
        else: new_b.append(jfeature)
    return new_b
    
def unify_bundle(b, d, ts=None): # bundle, dict, types template
    return Bundle(Feature(ts[i] if ts else f.type, d[f.val] if f.val in d else f.val) for (i, f) in enumerate(b))

def unify_grammar(g, d):
    return {left:unify_bundle(right, d) for (left, right) in g.items()}

def get_path(g, morphemes):
    dependents = []
    for (i, m) in enumerate(morphemes):
        is_first = True if i==0 else False
        is_last = True if i==len(morphemes)-1 else False
        dependents.extend(get_path_single(g[m], is_first, is_last)) 
    return tuple(dependents)
    
def get_path_single(li_syn, is_first=False, is_last=False):
    i_li_sel, li_sel = get_first_type(li_syn, is_sel)
    i_li_cat, li_cat = get_first_type(li_syn, is_cat)
    return li_syn[(i_li_sel if is_first else i_li_sel+1):i_li_cat] + li_syn[(i_li_cat if is_last else i_li_cat+1):]

tuple_without = lambda seq, m: tuple(x for x in seq if x != m)
tuple_without_multiple = lambda seq, ms: tuple(x for x in seq if not x in ms)

def concat_word(word, solution): # given a sequence of morphemes and a solution dict, build the word
    return "".join(solution[m] for m in word)
    
def cycle_check(eqs): # basic check to ensure no morpheme occurs twice in the same word
    return all(len(set(morphemes)) == len(morphemes) for morphemes in eqs.values())

def chimera_check(cc, f_out, f_in, g, eqs, solution): # use word graph to compare possible words with original words
    old_paths = set((concat_word(morphemes, solution), get_path(g, morphemes)) for word, morphemes in eqs.items())
    f_new = unify_name((f_out, f_in), id=f_in) # this is a temporary name
    new_g = unify_grammar({li_mor:li_syn for li_mor, li_syn in g.items() if li_mor != cc}, {f_out:f_new, f_in:f_new})    
    G, edge_labels = word_graph(new_g)
    
    try:
        nx.find_cycle(G)
        return False # fail the check on finding a cycle
    except nx.exception.NetworkXNoCycle: pass
    
    new_paths = set()
    for path in nx.all_simple_paths(G, source=nstart, target=nend):
        for morphemes in product(*[edge_labels[(path[i], path[i+1])] for i in range (0, len(path)-2)]):
            new_paths.add((concat_word(morphemes, solution), get_path(new_g, morphemes)))
            
    # TODO: also recognize bad syntactic paths beyond words (i.e. weird complements)
    
    return new_paths.issubset(old_paths)
    
def key_mor_sel(b):
    b_first = b[0]
    return (unify_type([b_first.type,]), b_first.val) if b_first.type in [ssel, wsel] else None
    
def key_postcat(b):
    cat_i = next(i for (i,f) in enumerate(b) if f.type == cat)
    return b[cat_i:]

def partition(elements, key_fun):
    d = {}
    for e in elements:
        dict_append(d, key_fun(e), e)
    return d.values()

def select_morphemes(g, ord, eqs, solution, is_high, is_suffix): # groups morphemes by msel or by cat & lees
    part_fun = key_postcat if is_high else key_mor_sel
    all_morphemes = []
    for li_mor in ord:
        all_morphemes.append((li_mor, g[li_mor], solution[li_mor]))
    return partition(all_morphemes, lambda x: part_fun(x[1]))

enumerate_neg = lambda l: ((i, l[i]) for i in range(-1, -len(l)-1, -1))
enum_bundle = lambda b, is_high: list(enumerate_neg(b) if is_high else enumerate(b))

splittable_syn = lambda x: (splittable_bundle(x[1]), False)
splittable_mor = lambda x: ([(char, True) for char in x[2]], True)
    
def get_batches(morphemes, is_high, is_suffix, batches):
    syn_splits = trie_splits(morphemes, is_high, splittable_syn, not(is_high))
    metabatches = []
    
    for syn_lis, syn_inds in syn_splits.items():

        i_right = next((i for (i, f) in reversed(enum_bundle(syn_lis[0][1], is_high)) if f.type in {lsel, lic}), None)
        if i_right: syn_inds = set(ind for ind in syn_inds if ind <= i_right) # preserving linear order; quick solution
        
        if syn_inds:
            sub_mor_splits = trie_splits(syn_lis, is_suffix, splittable_mor, True)
            
            for metabatch in partition(sub_mor_splits.keys(), lambda l: tuple(sorted([x[1] for x in l]))):
                if sum(len(batch) for batch in metabatch) > 1: # at least 2 LIs in metabatch
                    metabatches.append((metabatch, syn_inds))
                    if len(metabatch) > 1: # at least 2 batches in metabatch
                        metabatches.extend([([batch,], syn_inds) for batch in metabatch if len(batch) > 1])
       
    return metabatches

def note_decompose(batch_tuple):
    note_lis = ", ".join(["{} {}".format(li[2], pf(li[1])) for li in batch_tuple])
    note = "\ndecomposition: {}".format(note_lis)
    return note

def qdecompose(h, new_queue, qparams):    
    g, eqs, ord = transform_mg.results[h].mg, transform_mg.results[h].eqs, transform_mg.results[h].order
    solution = transform_mg.results[h].solution
    
    for is_high, is_suffix in qparams:
        for v in select_morphemes(g, ord, eqs, solution, is_high, is_suffix): # resulting batches don't overlap
            for metabatch, syn_inds in get_batches(v, is_high, is_suffix, {}):
                for syn_ind in syn_inds:
                    
                    new_cfg = transform_mg.results[h].cfg
                    new_g, new_solution, new_ord = dict(g), dict(solution), list(ord)
                    new_eqs, dec_dict = {}, {}
                    note = ""
                    
                    for i, batch in enumerate(metabatch):
                        
                        all_mors, all_syns, all_phons = zip(*batch)
                        shared_phon = lcs(all_phons) if is_suffix else lcp(all_phons)
                        
                        cat_i = temp_b_i(i) # separate category for each batch within a metabatch
                        name_i = (emp_mor, cat_i)
                        syn_i = unify_shared(all_syns, syn_ind, is_high)
                        bundle_i = Bundle(attach_shared(syn_i, Feature(ssel if is_high else cat, cat_i), is_high))
                        new_g[name_i] = bundle_i
                        new_solution[name_i] = shared_phon
                        
                        name_dict = {}
                        for j, (li_mor, li_feat, li_phon) in enumerate(batch):
                            name_i_j = (emp_mor, temp_nb_i_j(i)(j))
                            name_dict[li_mor] = name_i_j
                            syn_i_j = get_remaining(li_feat, syn_ind, is_high)
                            bundle_i_j = Bundle(attach_remaining(syn_i_j, Feature(cat if is_high else ssel, cat_i), is_high))
                            
                            new_g.pop(li_mor) # li_mor should always be available
                            new_g[name_i_j] = bundle_i_j
                            old_phon = new_solution.pop(li_mor)
                            new_solution[name_i_j] = old_phon[:len(old_phon)-len(shared_phon)] if is_suffix else old_phon[len(shared_phon):]
                            
                            replace_seq = [name_i_j, name_i] if is_high else [name_i, name_i_j]
                            dec_dict[li_mor] = replace_seq
                            
                            old_li = (li_mor, g[li_mor])
                            if is_high: upper_li, lower_li, cfg_ind = (name_i, bundle_i), (name_i_j, bundle_i_j), len(syn_i_j)
                            else: upper_li, lower_li, cfg_ind = (name_i_j, bundle_i_j), (name_i, bundle_i),syn_ind
                            new_cfg = cfg_decompose(new_cfg, old_li, upper_li, lower_li, cfg_ind) if new_cfg else None        

                        ord_first = next(i for i, mor in enumerate(new_ord) if mor in all_mors) + (0 if is_high else 1)
                        new_ord.insert(ord_first, name_i) # stick batch morpheme before/after first nonbatch morpheme
                        new_ord = [name_dict[mor] if mor in name_dict else mor for mor in new_ord] # replace names

                        note += note_decompose(batch)

                    for word, morphemes in eqs.items():
                        new_morphemes = []
                        for m in morphemes:
                            try: new_morphemes.extend(dec_dict[m])
                            except KeyError: new_morphemes.append(m)
                            new_eqs[word] = new_morphemes
                        
                    new_queue = add_step(h, new_g, new_ord, new_eqs, new_solution, new_cfg, new_queue, note)

    return new_queue
    
is_cat_changer = lambda li_syn, li_phon: len(li_syn) == 2 and is_msel(li_syn[0]) and li_phon == emp_true

def cat_changers(lis, mor_to_phon, changers_only): # assuming no identical edges
    li_dict, stop_pairs = {}, set()
    
    for li_mor, li_syn in lis:
        li_cat = get_cat(li_syn)
        if is_cat_changer(li_syn, mor_to_phon[li_mor]):
            li_dict[(li_syn[0].val, li_cat)] = li_mor # (v1, v2): morpheme name
        elif changers_only: # only allow the item if ALL (v1, v2) edges are category changers
            for v in [f.val for f in li_syn if is_sel(f)]:
                stop_pairs.add((v, li_cat))

    return {p:mor for p, mor in li_dict.items() if not p in stop_pairs}
    
def qcontract_single(h, new_queue, qparams):
        
    g, eqs, ord = transform_mg.results[h].mg, transform_mg.results[h].eqs, transform_mg.results[h].order
    solution = transform_mg.results[h].solution

    for ((li_orig, li_dest), li_mor) in cat_changers(g.items(), solution, True).items():

        cond = chimera_check(li_mor, li_orig, li_dest, g, eqs, solution) if use_chimera else True
        if cond:

            new_g, new_eqs = dict(g), dict(eqs)
            new_cfg = transform_mg.results[h].cfg
            del new_g[li_mor]

            li_new = unify_name((li_orig, li_dest))
            unify_dict = {li_orig:li_new, li_dest:li_new}
            new_g = unify_grammar(new_g, unify_dict)
            new_eqs = {word:tuple_without(morphemes, li_mor) for word, morphemes in new_eqs.items()}
            new_ord = [mor for mor in ord if not mor == li_mor]
            new_cfg = cfg_contract(new_cfg, [li_mor,], unify_dict) if new_cfg else None
            new_solution = {mor:phon for mor, phon in solution.items() if not mor == li_mor}
            
            note = "\nsingle contraction: {} → {}".format(li_orig, li_dest)
            new_queue = add_step(h, new_g, new_ord, new_eqs, new_solution, new_cfg, new_queue, note)

    return new_queue
    
def qremove_greedy(h, new_queue, qparams):
    g, eqs, ord = transform_mg.results[h].mg, transform_mg.results[h].eqs, transform_mg.results[h].order
    
    li_dict = cat_changers(g.items(), transform_mg.results[h].solution, False)
    gr = nx.MultiDiGraph(list(li_dict.keys())) # form graph from list of edges
    to_delete = {}
    
    for pair, mor in li_dict.items():
        gr_without = gr.copy()
        gr_without.remove_edge(*pair)
        if nx.has_path(gr_without, *pair):
            to_delete[mor] = [] # mark morpheme for deletion
            gr = gr_without # keep graph up to date
    
    if to_delete: 
        new_g = {li_mor:li_syn for li_mor, li_syn in g.items() if not li_mor in to_delete}
        new_ord = [mor for mor in ord if not mor in to_delete]
        
        for pair, mor in li_dict.items():
            if mor in to_delete:
                alts = nx.all_simple_paths(gr, source=pair[0], target=pair[1])
                to_delete[mor] = [[(li_dict[(alt[i], alt[i+1])]) for i in range(len(alt)-1)] for alt in alts]
        
        to_delete_keys, to_delete_vals = zip(*list(to_delete.items()))
        
        for element in product(*to_delete_vals): # create a new step for each combination
            cross_g = new_g.copy()
            cross_ord = new_ord.copy()
            cross_eqs = {}
            cross_dict = dict(zip(to_delete_keys, element)) # associate replacement lists with original morphemes
            cross_solution = {li_mor:li_phon for li_mor, li_phon in transform_mg.results[h].solution.items() if not li_mor in to_delete}
                     
            for word, morphemes in eqs.items():
                new_morphemes = []
                for m in morphemes:
                    try: new_morphemes.extend(cross_dict[m])
                    except KeyError: new_morphemes.append(m)
                cross_eqs[word] = new_morphemes

            note_items = [(pf(g[mor]), ", ".join(["{}".format(pf(g[m])) for m in cross_dict[mor]])) for mor in cross_dict]
            note = "\nreplacement: {}".format("; ".join(["{} with {}".format(item[0], item[1]) for item in note_items]))
            
            cross_cfg = transform_mg.results[h].cfg # no need to copy, as the original grammar is not modified
            for mor, alt_mors in cross_dict.items():
                cross_cfg = cfg_remove(cross_cfg, (mor, g[mor]),  [(m, g[m]) for m in alt_mors]) if cross_cfg else None

            new_queue = add_step(h, cross_g, cross_ord, cross_eqs, cross_solution, cross_cfg, new_queue, note)
    
    return new_queue
    
def splittable_bundle(b): # True/False: "allowed/not allowed to split before this position"
    i_left = 1 if is_msel(b[0]) else 0
    i_right = get_first_type(b, is_cat)[0] + 1
    return [(f, True if i in range(i_left, i_right) else False) for (i, f) in enumerate(b)]
    
def items_to_trie(mg_items, rev, seq_fun):
    trie = {}
    for li in mg_items: # for each LI
        temp_dict = trie
        li_bundle, leaf_val = seq_fun(li)
        if rev: li_bundle.reverse()
        for f in li_bundle: # for each feature in the LI
            temp_dict = temp_dict.setdefault(f, {})
        temp_dict[li] = leaf_val # create leaves
    return trie

def trie_splits(trie_items, rev, seq_fun, split_root):
    trie = items_to_trie(trie_items, rev, seq_fun)
    trie_splits_aux.result = {}
    trie_splits_aux(trie, rev, 0, split_root) # None if we don't want a root batch
    return trie_splits_aux.result
    
def trie_splits_aux(t, rev, i, can_split_parent):
    t_yield = []
    t_split = set()
    for key in t: # process the children first
        if isinstance (t[key], dict): # nonleaf child
            key_yield = trie_splits_aux(t[key], rev, i-1 if rev else i+1, key[1])
            t_yield.extend(key_yield)
            t_split.add(key[1])
        else: # leaf child
            t_yield.append(key)
            t_split.add(t[key])
    can_split = can_split_parent == True if rev else (not False in t_split)
    if can_split and len(t_yield) > 1: # 0 to allow batches of one
        dict_append(trie_splits_aux.result, tuple(t_yield), i, True)
    return t_yield # list of leaves under t

get_first_type = lambda b, t: next(((i, f.val) for i, f in enumerate(b) if t(f)), (0, None))
terminal_exp = lambda b: (Chain(atomic, b),) # TODO: transfer to grammars.py and reuse there

def cfg_add(cfg, left, right, is_term, usage_li, usage_ind, usage_num):
    cfg.setdefault(left, {}) # if left is absent
    cfg[left].setdefault(right, Rule_data(is_term, {})) # if right is absent
    cfg[left][right].usage.setdefault(usage_li, LI_usage(usage_ind, 0)) # if usage_li is absent in rule usage
    cfg[left][right].usage[usage_li].num += usage_num
    return

def cfg_decompose(orig_cfg, old_li, upper_li, lower_li, ind):
    cfg = {}
    
    term_old, term_upper, term_lower = terminal_exp(old_li[1]), terminal_exp(upper_li[1]), terminal_exp(lower_li[1])
    term_old_uses = orig_cfg[term_old][(old_li[0],)].usage[old_li[0]].num
    
    cfg_add(cfg, term_upper, (upper_li[0],), True, upper_li[0], 0, term_old_uses) # add new upper LI
    
    for left in orig_cfg:
        for right in orig_cfg[left]:
            
            current_is_term = orig_cfg[left][right].is_term
            for usage_li, usage_data in orig_cfg[left][right].usage.items():
                if usage_li != old_li[0]: # adding all unrelated rule usage to the new cfg
                    cfg_add(cfg, left, right, current_is_term, usage_li, usage_data.ind, usage_data.num)

            if old_li[0] in orig_cfg[left][right].usage: # the rule is associated with old_li
                old_usage = orig_cfg[left][right].usage[old_li[0]] # get number of relevant uses

                if left == start_symbol or old_usage.ind > ind: # expand upper LI; mostly same rule but reassign uses to upper_li
                    upper_ind = old_usage.ind-(ind-1)
                    new_right_first = (Chain(derived, right[0][0].features),) + right[0][1:] # first argument is always derived
                    new_right = (new_right_first,) + right[1:]
                    cfg_add(cfg, left, new_right, False, upper_li[0], upper_ind, old_usage.num)
                    
                else: # expanding lower LI
                    if current_is_term: # add new terminal rule for lower LI
                        new_left = term_lower
                        new_right = (lower_li[0],)
                    else:
                        new_left = (Chain(left[0].type, lower_li[1][old_usage.ind:]),) + left[1:]
                        new_right = ((Chain(right[0][0].type, lower_li[1][old_usage.ind-1:]),) + right[0][1:],) + right[1:]
                    cfg_add(cfg, new_left, new_right, current_is_term, lower_li[0], old_usage.ind, old_usage.num)
                    
                    if old_usage.ind == ind: # merge upper and lower
                        merge_right = (term_upper, new_left)
                        cfg_add(cfg, left, merge_right, False, upper_li[0], 1, old_usage.num)
            
    return cfg
    
def cfg_contract(orig_cfg, old_names, unify_dict):
    cfg = {start_symbol:{}}
    
    for left in orig_cfg:
        new_left = left if left == start_symbol else tuple(Chain(c.type, unify_bundle(c.features, unify_dict)) for c in left)
        
        for right in orig_cfg[left]:
            if orig_cfg[left][right].is_term == True: new_right = right
            else: new_right = tuple(tuple(Chain(c.type, unify_bundle(c.features, unify_dict)) for c in r) for r in right)
        
            new_usage_keys = set(orig_cfg[left][right].usage.keys()).difference(old_names) 
            if new_usage_keys: # if the LHS expression can be headed by something other than the items we just deleted
                cfg.setdefault(new_left, {})
                cfg[new_left].setdefault(new_right, Rule_data(orig_cfg[left][right].is_term, {}, None))
                
                for li in new_usage_keys:
                    cfg[new_left][new_right].usage.setdefault(li, LI_usage(orig_cfg[left][right].usage[li].ind, 0))
                    if orig_cfg[left][right].usage[li].ind != cfg[new_left][new_right].usage[li].ind:
                        raise Exception(exception_ind)
                    cfg[new_left][new_right].usage[li].num += orig_cfg[left][right].usage[li].num
      
    cfg = rule_heads(closure(cfg, cfg_to_exps(cfg))) # run closure, add potential heads to new rules. Needed because licensees!
                
    return cfg
    
def cfg_remove(orig_cfg, old_li, new_lis):
    cfg = {}
    
    term_old = terminal_exp(old_li[1])
    term_old_uses = orig_cfg[term_old][(old_li[0],)].usage[old_li[0]].num
    terms_new = [terminal_exp(new_li[1]) for new_li in new_lis]
       
    for i in range(len(new_lis)):
        cfg_add(cfg, terms_new[i], (new_lis[i][0],), True, new_lis[i][0], 0, term_old_uses) # add new terminals
    
    for left in orig_cfg:
        for right in orig_cfg[left]:
            
            current_is_term = orig_cfg[left][right].is_term
            
            for usage_li, usage_data in orig_cfg[left][right].usage.items():
                if usage_li != old_li[0]: # add all usage not associated with old_li
                    cfg_add(cfg, left, right, current_is_term, usage_li, usage_data.ind, usage_data.num)
                  
            if old_li[0] in orig_cfg[left][right].usage and current_is_term == False:
                old_usage = orig_cfg[left][right].usage[old_li[0]] # get uses associated with old_li; ind should always be 1

                new_right_snd = right[1] # expression selected by old_li
                
                for i in range(len(new_lis)):
                    new_left = (Chain(derived, new_lis[i][1][1:]),) + right[1][1:]
                    new_right_fst = terms_new[i]
                    cfg_add(cfg, new_left, (new_right_fst, new_right_snd), False, new_lis[i][0], old_usage.ind, old_usage.num)
                    new_right_snd = new_left
    
    return cfg
    
def feedback_time(chunk_start, h_i, l):
    if (h_i) % 1 == 0 or h_i == l:
        print("Grammars processed: {} out of {}; time: {}".format(h_i+1, l, datetime.now()-chunk_start), end='\r')
        return datetime.now()
    else: return chunk_start
    
def feedback_level(fun, local_mdls, l, len_orig, len_new, len_all):
    best_overall = min(local_mdls.values(), key=hsize_aux)
    print("Level {}: {}. Known grammars: {}, best so far: ({:0.2f}, {:0.2f}). Processed: {}, new: {}, in queue: {}".format(l, fun.__name__, len(local_mdls), *best_overall, len_orig, len_new, len_all))
    
def feedback_cycle(local_results, local_mdls, current_best, new_best, queue_len, bestx=1):
    lcb = len(current_best)
    unchanged = next((i for i in range(lcb) if new_best[i] != current_best[i]), lcb)
    print("Cycle completed. No change in top {}/{}; in queue: {}; grammars recorded: {}\n".format(unchanged, check_top, queue_len, len(local_results)))
    
    if verbose_feedback:
        for pos in range(0, bestx):
            try:
                res = local_results[new_best[pos]]
                print("#{} grammar: {}; ({:0.2f}, {:0.2f}); obtained at level {}, {}".format(pos, new_best[pos], *local_mdls[new_best[pos]], res.level, res.note))
                pretty_mg(step_to_mg(res.mg, res.solution))
                pretty_cfg(res.cfg, split_lex, nonzero=True, used=True)
                print("-------")
            except IndexError: break
        print()
        
def get_history(h, local_results):
    history, prev_hash = [], h
    while prev_hash:
        history.insert(0, prev_hash)
        prev_hash = local_results[history[0]].parent
    return history

def feedback_history(best_hash, local_results, local_mdls, plot_path):
    print("Recording grammar history...")
        
    for history_hash in get_history(best_hash, local_results):
        res = local_results[history_hash]
        
        if verbose_feedback:
            print("Obtained at level: {}, hash: {}, cost: ({:0.2f}, {:0.2f}), rank: {}; {}".format(res.level, history_hash, local_mdls[history_hash][0], local_mdls[history_hash][1], res.rank, res.note))
            pretty_mg(step_to_mg(res.mg, res.solution))
            pretty_mg(res.mg)
            pretty_eqs(res.eqs)
            pretty_cfg(res.cfg, split_lex, nonzero=True, used=True)
        
        if plot_path != None:
            plot_name = ["L{}".format(res.level), "({:0.2f}, {:0.2f})".format(local_mdls[history_hash][0], local_mdls[history_hash][1]), "{}".format(res.rank)]
            plot_dir = os.path.join(os.getcwd(), r'{}'.format(plot_path))        
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)            
            mg_to_plot(step_to_mg(res.mg, res.solution), "{}/{}".format(plot_dir, "-".join(plot_name)), res.note, res.fresh)
    print()
    
def apply_fun_cycle(fun, queue, qparams, cutoff):
    queue_out = set(queue)
    while True:
        queue = apply_fun(fun, queue, qparams, cutoff, keep_orig=False) # only steps that JUST underwent contraction
        queue_out.update(queue)
        if not queue: return queue_out
    
def apply_fun(fun, queue, qparams, cutoff, keep_orig=True):
    queue_out = set(queue) if keep_orig else set() # initialize output queue with or without original grammars
    len_orig = len(queue_out)
    
    transform_mg.lcounter +=1
    chunk_start = datetime.now()
    for h_i, h in enumerate(queue): # process each grammar in the queue
        queue_out = fun(h, queue_out, qparams)
        chunk_start = feedback_time(chunk_start, h_i, len(queue))       

    queue_out = queue_out.difference(transform_mg.processed) # discard already processed grammars
    queue_sorted_overall = sorted(queue_out, key=lambda h: hsize(transform_mg.mdls)(h))
    
    if cutoff != None:
         queue_final = set()
         if queue_out:
             cutoff_hash = queue_sorted_overall[cutoff] if len(queue_sorted_overall) > cutoff else queue_sorted_overall[-1]
             cutoff_val = hsize(transform_mg.mdls)(cutoff_hash)
             cutoff_ind = next((i for (i, v) in enumerate(queue_sorted_overall) if hsize(transform_mg.mdls)(v) > cutoff_val), None)
             queue_final.update(queue_sorted_overall[:cutoff_ind])
    
    else: queue_final = queue_out
    
    for queue_hash in queue_final:
        transform_mg.results[queue_hash].rank = (queue_sorted_overall.index(queue_hash))
    
    feedback_level(fun, transform_mg.mdls, transform_mg.lcounter, len(queue), len(queue_out)-len_orig, len(queue_final))
    return queue_final
    
def step_checks(new_g, new_eqs, new_ord, new_cfg, note):
    if set(new_ord) == set(new_g.keys()): pass
    else: raise Exception("{}\nOrder does not match MG!".format(note))
    
    if new_cfg == None or cfg_check(new_cfg, new_eqs): pass
    else:
        pretty_mg(new_g)
        pretty_eqs(new_eqs)
        print("========")
        cfg_check(new_cfg, new_eqs, True)
        print()
        pretty_cfg(new_cfg)
        raise Exception("{}\nCFG usage values don't check out!".format(note))    
    
def add_step(h, new_g, new_ord, new_eqs, new_solution, new_cfg, new_queue, note=""):
    
    step_checks(new_g, new_eqs, new_ord, new_cfg, note)    
    new_step = make_step(new_g, new_ord, new_eqs, new_solution, new_cfg)
    new_hash = hash_step(new_step.mg, new_step.eqs)
    
    if not new_hash in transform_mg.mdls:
        new_queue.add(new_hash)
        transform_mg.mdls[new_hash] = cost_function(step_to_mg(new_step.mg, new_step.solution), new_step.cfg)
        new_step.parent, new_step.level, new_step.note = h, transform_mg.lcounter, note
        transform_mg.results[new_hash] = new_step

    return new_queue
    
def transform_mg(orig_mg, orig_eqs, orig_solution, orig_cfg, beam_size):
    
    global orig_names
    orig_names = get_feature_names(orig_mg) # record feature names used in the input MG

    transform_mg.lcounter = -1
    transform_mg.results, transform_mg.mdls, transform_mg.processed = {}, {}, set()
    ord = list(sorted(orig_mg.keys(), key=lambda x:[int(i) for i in x[1:]]))
    queue = add_step(None, orig_mg, ord, orig_eqs, orig_solution, orig_cfg, set(), "original")
    
    new_best = [] # initialize best hash list
    while True:
        
        queue = apply_fun(qdecompose, queue, qparams, beam_size)
        queue = apply_fun_cycle(qcontract_single, queue, qparams, beam_size)
        queue = apply_fun(qremove_greedy, queue, qparams, beam_size)
        
        transform_mg.processed.update(queue) # keep track of grammars that have already been in the queue
        mdl_hashes_sorted = sorted(transform_mg.mdls.keys(), key=hsize(transform_mg.mdls))
        current_best, new_best = new_best, mdl_hashes_sorted[:check_top]
        
        valuable_hashes = set(get_history(mdl_hashes_sorted[0], transform_mg.results))
        for h in queue: valuable_hashes.update(get_history(h, transform_mg.results))        
        transform_mg.results = {h:transform_mg.results[h] for h in valuable_hashes}
        
        feedback_cycle(transform_mg.results, transform_mg.mdls, current_best, new_best, len(queue))
        
        if len(queue) == 0 or new_best == current_best: break
        
    best_hash, best_mdl = min(transform_mg.mdls.items(), key=lambda x: hsize_aux(x[1]))
    if verbose_history: feedback_history(best_hash, transform_mg.results, transform_mg.mdls, level_plot_path)    
    return transform_mg.results[best_hash], best_mdl
    
parser = argparse.ArgumentParser()
parser.add_argument('-vf', '--verbose_feedback', action='store_true', default=False, help='print transformation steps')
parser.add_argument('-vh', '--verbose_history', action='store_true', default=False, help='print best grammar history')
parser.add_argument('-c', '--corpus', action='store', nargs='?', type=str, help='corpus name')
parser.add_argument('-cs', '--corpus_size', action='store', nargs='?', type=int, default=None, help='corpus size')
parser.add_argument('-gn', '--generate_new', action='store_true', default=False, help='force generate new corpus')
parser.add_argument('-go', '--generate_only', action='store_true', default=False, help='generate new corpus and stop')
parser.add_argument('-gm', '--generate_method', action='store', nargs='?', type=str, default="gen_rand", help='corpus generation method')
parser.add_argument('-ct', '--check_top', action='store', nargs='?', type=int, default=50, help='number of grammars for the stop criterion. Default: 50')
parser.add_argument('-bs', '--beam_size', action='store', nargs='?', type=int, default=100, help='number of candidates to keep at each level. Default: 100')
parser.add_argument('-gc', '--grammar_cost', action='store', nargs='?', type=str, default="mdl_1d", help='grammar cost function')
parser.add_argument('-cc', '--corpus_cost', action='store', nargs='?', type=str, default="mdl_cfg", help='corpus cost function')
parser.add_argument('-oc', '--overall_cost', action='store', nargs='?', type=str, default="hsize_ord", help='overall cost function. Values: hsize_ord, hsize_sum, hsize_grammar, hsize_grammar_alt')
parser.add_argument('-noch', '--nocheck', action='store_true', default=False, help='suppress the chimera_check heuristic')
args = parser.parse_args()

verbose_feedback = args.verbose_feedback
verbose_history = args.verbose_history
corpus_name = args.corpus
gen_new = args.generate_new
gen_only = args.generate_only
gen_method = eval(args.generate_method)
corpus_size = args.corpus_size if args.corpus_size != None else 100 if gen_method==gen_rand else inf

qparams = [(False, False), (True, True)] # list of of pairs: (is_high, is_suffix); simultaneously find roots at start and suffixes

beam_size = args.beam_size
check_top = args.check_top
use_chimera = not(args.nocheck)
grammar_cost = eval(args.grammar_cost)
corpus_cost = eval(args.corpus_cost)
cost_function = mdl_full(grammar_cost, corpus_cost)
split_lex = True if corpus_cost == mdl_cfg_split else False
hsize_aux = eval(args.overall_cost)
hsize = lambda d: (lambda h: hsize_aux(d[h]))

def show_examples(examples, mcfg, show_all):
    if show_all or len(examples) <= 50:
        for ex in examples: pretty_sentence(mcfg_string(ex, mcfg))
    else:
        for ex in examples[:25]: pretty_sentence(mcfg_string(ex, mcfg))
        print("...")
        for ex in examples[-25:]: pretty_sentence(mcfg_string(ex, mcfg))
    print("Total examples: {}".format(len(examples)))

if __name__ == "__main__":
    
    start_time = datetime.now()
    
    args_data = "{}_bs{}_{}_{}".format(corpus_name, beam_size, args.overall_cost, "ch" if use_chimera else "noch")
    # args_data = "{}-{}-{}-bs{}-{}-{}-{}-{}".format(corpus_name, args.generate_method, corpus_size, beam_size, args.grammar_cost, args.corpus_cost, args.overall_cost, "ch" if use_chimera else "noch")
    level_plot_path = "plots/{}".format(args_data)
    corpus_path = "corpora/{}_{}_{}".format(corpus_name, args.generate_method, corpus_size)
    
    if verbose_history and os.path.exists(level_plot_path): shutil.rmtree(level_plot_path)
    
    global head_name
    
    if gen_new or not os.path.exists("{}.db".format(corpus_path)): # generate a new corpus 
        print("Generating corpus:")
        start_mg, head_name = file_to_mg(corpus_name)
        orig_mg, orig_eqs, mor_to_str = preprocess_mg(start_mg)
        
        mcfg, examples, orig_eqs = make_corpus(orig_mg, orig_eqs, mor_to_str, corpus_size, gen_method, start_exp_fun(head_name))
        
        shelf = shelve.open(corpus_path)
        shelf['head_name'], shelf['examples'], shelf['mcfg'], shelf['mg'], shelf['eqs'] = head_name, examples, mcfg, orig_mg, orig_eqs
        shelf['solution'] = mor_to_str
        shelf.close()
        
    else: # load an existing corpus
        print("Loading corpus:")
        shelf = shelve.open(corpus_path)
        head_name, examples, mcfg, orig_mg, orig_eqs = shelf['head_name'], shelf['examples'], shelf['mcfg'], shelf['mg'], shelf['eqs']
        mor_to_str = shelf['solution']
        shelf.close()
    
    show_examples(examples, mcfg, gen_only)
    orig_solved = step_to_mg(orig_mg, mor_to_str)
    print("Input cost: ({:0.2f}, {:0.2f})\n".format(*cost_function(orig_solved, mcfg)))
    mg_to_plot(orig_solved, "plots/{}_original".format(corpus_name))
        
    if not gen_only:  
        
        orig_cfg = drop_maps(mcfg) if hsize_aux != hsize_grammar else None
        pretty_cfg(orig_cfg, nonzero=True, used=True)
        
        best_result, best_cost = transform_mg(orig_mg, orig_eqs, mor_to_str, orig_cfg, beam_size)
        
        print()
        new_solved = step_to_mg(best_result.mg, best_result.solution)
        pretty_mg(new_solved, True)
        pretty_eqs(best_result.eqs)
        mg_to_plot(new_solved, "plots/{}_({:0.2f}, {:0.2f})".format(args_data, best_cost[0], best_cost[1]))
        print("Output cost: {}".format(cost_function(new_solved, best_result.cfg)))        
        print("Best grammar hash: {}".format(hash_step(new_solved)))
        
        mg_to_file(new_solved, "{}_out".format(corpus_name), head_name)
    
    print("Elapsed time: {}\n".format(datetime.now() - start_time))
