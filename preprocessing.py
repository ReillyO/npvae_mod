from __future__ import print_function
from model.utils import *
from rdkit import Chem
from collections import Counter
import numpy as np
import torch
import pickle
import timeit

def write_verbose_out(string):
  with open("preproc.v.txt", 'a') as f:
    f.write(string + "\n")

def main(args):
    write_verbose_out("entered main")
    process_times = []
    all_smiles = []
    # changing loading to probe memory overuse - oo
    with open(args.smiles_path) as f:
        for line in f:
            all_smiles.append(line.strip("\r\n ").split()[0])
    print("Number of SMILES entered: ", len(all_smiles))
    print("Preprocessing consists of nine processes")
    write_verbose_out("processed smiles")
    
    tic = timeit.default_timer()

    print("Process 1/9 is running", end = '...')
    write_verbose_out("process 1")
    mols = []
    cou = 0
    molcount = 1
    chunkcount = 0
    for i in range(len(all_smiles)):
        try:
            mol = Chem.MolFromSmiles(all_smiles[i])
            mol = sanitize(mol, kekulize = False)
            mol = Chem.RemoveHs(mol)
            mols.append(mol)
        except:
            cou += 1
        #if molcount % 10000 == 0:
        #  with open("chunk"+str(chunkcount)+".dat", 'w') as chunkf:
        #    chunkf.write(str(mols))
        #    mols.clear()
        #    chunkcount += 1
        molcount += 1
    if cou > 0:
        raise ValueError("There might be some errors. Check your SMILES data.")
    print('done')
    toc = timeit.default_timer()
    process_times.append(toc - tic)
    
    tic = timeit.default_timer()
    print("Process 2/9 is running", end = '...')
    write_verbose_out("Process 2/9 is running")
    count_labels = [] #(substructureSMILES,(AtomIdx in substructure, join order)xN)->frequency of use of label
    fragments = []
    for i, m in enumerate(mols):
        cl, frag = count_fragments(m)
        #print(cl)
        #print(frag)
        count_labels += cl
        fragments.append(frag)
    count_labels = Counter(count_labels)
    print('done')
    toc = timeit.default_timer()
    process_times.append(toc - tic)     

    tic = timeit.default_timer()
    print("Process 3/9 is running", end = '...')
    write_verbose_out("process 3")
    mapidx_list = []
    labelmap_dict_list = []
    bondtype_list = []
    max_mapnum_list = []
    fragments_list = []
    fragments = []
    for i, m in enumerate(mols):
        mapidxs, labelmap_dict, bondtypes, max_mapnums, frag = find_fragments(m, count_labels, args.frequency)
        # mapidxs: list of pairs of fragment SMILES and their index in the labelmap_dict (?)
        # labelmap_dict: dictionary where key is fragment SMILES and entry is a dict of (AtomIdx, sep_idx) pairs
        # bondtypes: simple list of RDKit BondTypes 
        # max_mapnums: number of fragmentations performed in the molecule
        # fragments: list of SMILES strings of fragments generated in this process
        mapidx_list.append(mapidxs)
        labelmap_dict_list.append(labelmap_dict)
        bondtype_list.append(bondtypes)
        max_mapnum_list.append(max_mapnums)
    print('done')
    toc = timeit.default_timer()
    process_times.append(toc - tic)
    
    del mols # freeing up memory space? - oo
    del max_mapnum_list
    
    
    tic = timeit.default_timer()        
    print("Process 4/9 is running", end = '...')
    write_verbose_out("process 4")
    rev_labelmap_dict_list = []
    for lm_d in labelmap_dict_list: # for every map in the map list
        rev_labelmap_dict, deg = revise_maps(lm_d) # finds maximum degree and updates connectivity map so it starts from 1 
        rev_labelmap_dict_list.append(rev_labelmap_dict)
    print('done')
    toc = timeit.default_timer()
    process_times.append(toc - tic)

    tic = timeit.default_timer()
    print("Process 5/9 is running", end = '...')
    write_verbose_out("process 5")
    labels = []
    for ld in rev_labelmap_dict_list: # for every label dictionary in the list
        for k in ld.keys(): # every fragment in the label dict
            for l in ld[k]: # every (AtomIdx, sep_id) in the fragment dict
                label = [] 
                label.append(k) # append fragment
                label.append(l) # append (AtomIdx, sep_id)
                if label not in labels:
                    labels.append(label) # add to labels list if not already present
    print('done')
    toc = timeit.default_timer()
    process_times.append(toc - tic)

    tic = timeit.default_timer()    
    print("Process 6/9 is running", end = '...')
    write_verbose_out("process 6")
    graphs = []
    sub_trees = []
    root_ans_list = []
    label_ans_list = []
    bond_ans_list = []
    for i in range(len(mapidx_list)):
        if i % 50000 == 0: write_verbose_out(str(i)+" molecules")
        g, sub_tree, root_answer, l_ans_list, b_ans_list = make_graph(mapidx_list[i], labelmap_dict_list[i], rev_labelmap_dict_list[i] ,labels, bondtype_list[i])
        graphs.append(g)
        sub_trees.append(sub_tree)
        root_ans_list.append(root_answer)
        label_ans_list.append(l_ans_list)
        bond_ans_list.append(b_ans_list)
    print('done')
    toc = timeit.default_timer()
    process_times.append(toc - tic)

    tic = timeit.default_timer()    
    print("Process 7/9 is running", end = '...')
    write_verbose_out("process 7")
    l_1_counter = np.zeros(len(labels), dtype = int)
    l_counter = np.zeros(len(labels), dtype = int)
    b_counter = np.zeros(3, dtype = int)
    t_counter = np.zeros(2, dtype = int)
    bg_node_list = []
    target_id_list = []
    topo_ans_list = []
    for i in range(len(graphs)):
        if i % 50000 == 0: write_verbose_out(str(i)+" graphs")
        _, bg_node_l, target_id_l, topo_ans_l, \
        l_1_counter, l_counter, b_counter, t_counter = \
        demon_decoder(graphs[i], sub_trees[i], \
                      root_ans_list[i], label_ans_list[i], bond_ans_list[i], \
                      l_1_counter, l_counter, b_counter, t_counter, \
                      labels)
        bg_node_list.append(bg_node_l)
        target_id_list.append(target_id_l)
        topo_ans_list.append(topo_ans_l)
    print('done')
    toc = timeit.default_timer()
    process_times.append(toc - tic)

    tic = timeit.default_timer()    
    print("Process 8/9 is running", end = '...')
    write_verbose_out("process 8")
    #Creating weights for cross-entropy
    t_wei = []
    b_wei = []
    l_wei = []
    l_1_wei = []
    t_max = np.max(t_counter)
    b_max = np.max(b_counter)
    l_max = np.max(l_counter)
    l_1_max = np.max(l_1_counter)
    for i in range(len(t_counter)):
        t_wei.append(t_max / (t_counter[i] + 1e-7))
    for i in range(len(b_counter)):
        b_wei.append(b_max / (b_counter[i] + 1e-7))
    for i in range(len(l_counter)):
        l_wei.append(l_max / (l_counter[i] + 1e-7))
        l_1_wei.append(l_1_max / (l_1_counter[i] + 1e-7))
    t_wei = np.array(t_wei)
    b_wei = np.array(b_wei)
    l_wei = np.array(l_wei)
    l_1_wei = np.array(l_1_wei)
    t_weights = torch.from_numpy(t_wei).float()
    b_weights = torch.from_numpy(b_wei).float()
    l_weights = torch.from_numpy(l_wei).float()
    l_1_weights = torch.from_numpy(l_1_wei).float()
    print('done')
    toc = timeit.default_timer()
    process_times.append(toc - tic)

    tic = timeit.default_timer()    
    print("Process 9/9 is running", end = '...')
    write_verbose_out("process_9")
    ecfp_list3D = []
    for i in range(len(all_smiles)):
        if i % 50000 == 0: write_verbose_out(str(i)+" molecules")
        mol_ecfp = make_ecfp3D(all_smiles[i], args.fpbit, args.radius)
        mol_ecfp = np.asarray(mol_ecfp)
        mol_ecfp = torch.from_numpy(mol_ecfp).float()
        mol_ecfp = mol_ecfp.unsqueeze(0)
        ecfp_list3D.append(mol_ecfp)
    m_counter = np.zeros(args.fpbit, dtype = int)
    for fp in ecfp_list3D:
        for i, bit in enumerate(fp[0]):
            if bit:
                m_counter[i] += 1
    m_wei = []
    m_max = np.max(m_counter)
    for i in range(len((m_counter))):
        m_wei.append(m_max / (m_counter[i] + 1e-7))
    m_wei = np.array(m_wei)
    m_weights = torch.from_numpy(m_wei).float()
    print('done')
    toc = timeit.default_timer()
    process_times.append(toc - tic)

    tic = timeit.default_timer()    
    print("Saving the created data to the specified path", end = '...')
    #Writing the created lists
    with open(args.save_path + "/input_data/graphs", "wb")as f:
        pickle.dump(graphs, f)
    with open(args.save_path + "/input_data/sub_trees", "wb")as f:
        pickle.dump(sub_trees, f)
    with open(args.save_path + '/input_data/labels', "wb") as f:
        pickle.dump(labels, f)
    with open(args.save_path + "/input_data/weights/t_weights", "wb")as f:
        pickle.dump(t_weights, f)
    with open(args.save_path + "/input_data/weights/b_weights", "wb")as f:
        pickle.dump(b_weights, f)
    with open(args.save_path + "/input_data/weights/l_weights", "wb")as f:
        pickle.dump(l_weights, f)
    with open(args.save_path + "/input_data/weights/l_1_weights", "wb")as f:
        pickle.dump(l_1_weights, f)
    with open(args.save_path + "/input_data/ecfp_list3D", "wb")as f:
        pickle.dump(ecfp_list3D, f)
    with open(args.save_path + "/input_data/weights/m_weights", "wb")as f:
        pickle.dump(m_weights, f)
    with open(args.save_path + "/input_data/bg_node_list", "wb")as f:
        pickle.dump(bg_node_list, f)
    with open(args.save_path + "/input_data/root_ans_list", "wb")as f:
        pickle.dump(root_ans_list, f)
    with open(args.save_path + "/input_data/label_ans_list", "wb")as f:
        pickle.dump(label_ans_list, f)
    with open(args.save_path + "/input_data/bond_ans_list", "wb")as f:
        pickle.dump(bond_ans_list, f)
    with open(args.save_path + "/input_data/topo_ans_list", "wb")as f:
        pickle.dump(topo_ans_list, f)
    with open(args.save_path + "/input_data/target_id_list", "wb")as f:
        pickle.dump(target_id_list, f)
    toc = timeit.default_timer()
    process_times.append(toc - tic)
    with open(args.save_path + "process_times.txt", 'w') as f:
        for time in process_times: f.write(str(time) + '\n')

    print('done in ' + str(sum(process_times)) + ' time')

if __name__ == "__main__":
    print("started")
    import warnings
    import argparse
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type = str,
                        default = "./smiles_data/drugbank_smiles.txt", help = "Path of SMILES data for input compounds (delete SMILES containing '.')")
    parser.add_argument("-freq", "--frequency", type = int, default = 5,
                        help = "Threshold frequencies at decomposition")
    parser.add_argument("-fpbit", type = int, default = 2048,
                        help = "Number of bits of ECFP")
    parser.add_argument("-r", "--radius", type = int, default = 2,
                        help = "Effective radius of ECFP")
    parser.add_argument("--save_path", type = str,
                        default = "./save_data", help = "Path to save created data")
    args = parser.parse_args()
    print("parsed args")
    main(args)
