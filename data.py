import torch
torch.manual_seed(0)
from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor, Lambda
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils import calculate_class_weights
import pandas as pd
import random
import yaml
from utils import *


class LocalizationDataset(Dataset):
    def __init__(self, samples, configs):
        self.samples = samples
        self.n = configs.encoder.num_classes
        print(self.count_samples_by_class(self.n, self.samples))
        self.class_weights = calculate_class_weights(self.count_samples_by_class(self.n, self.samples))
        print(self.class_weights)
    @staticmethod
    def count_samples_by_class(n, samples):
        """Count the number of samples for each class."""
        class_counts = np.zeros(n)  # one extra is for samples without motif
        # Iterate over the samples
        for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
            class_counts += type_protein
        return class_counts
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        id, id_frag_list, seq_frag_list, target_frag_list, type_protein = self.samples[idx]

        labels = np.where(type_protein == 1)[0]
        weights = []
        print(labels)
        for label in labels:
            weights.append(self.class_weights[label])
        sample_weight = max(weights)

        type_protein = torch.from_numpy(type_protein)
        return id, id_frag_list, seq_frag_list, target_frag_list, type_protein, sample_weight 


class SupconDataset(Dataset):
    # prot_id, id_frag_list, seq_frag_list, target_frag_list, type_protein
    def __init__(self, samples, configs):
        # try not to change the input format of LocalizationDataset
        self.samples = samples
        self.configs = configs
        label2idx = {"Nucleus": 0, "ER": 1, "Peroxisome": 2, "Mitochondrion": 3, "Nucleus_export": 4,
                     "SIGNAL": 5, "chloroplast": 6, "Thylakoid": 7}
        idx2label = {v: k for k, v in label2idx.items()}

        simplified_samples = [(id_frag_list, seq_frag_list, type_protein) for _, id_frag_list, seq_frag_list, __, type_protein in samples]

        converted_samples = []
        for id_frag_list, seq_frag_list, type_protein in simplified_samples:
            labels = [idx2label[i] for i, val in enumerate(type_protein) if val == 1]
            converted_samples.append((id_frag_list, seq_frag_list, labels))

        extended_converted_samples = []
        for id_frag_list, seq_frag_list, labels in converted_samples:
            if len(id_frag_list) != len(seq_frag_list):
                raise ValueError("Length of id_frag_list and seq_frag_list does not match.")
            length = len(id_frag_list)
            extended_labels = [labels for _ in range(length)]
            extended_converted_samples.append((id_frag_list, seq_frag_list, extended_labels))

        flattened_samples = []
        for id_frag_list, seq_frag_list, extended_labels in extended_converted_samples:
            for i in range(len(id_frag_list)):
                labels = extended_labels[0]
                flattened_samples.append((id_frag_list[i], seq_frag_list[i], labels))
        self.id_seq_label = flattened_samples

        self.id_label = {item[0]: item[2] for item in self.id_seq_label}
        self.label_id = {}
        for item in self.id_seq_label:
            id, seq, labels = item
            for label in labels:
                if label not in self.label_id:
                    self.label_id[label] = [id]
                else:
                    self.label_id[label].append(id)

        self.full_list = []
        self.mine_neg = None  # hard mining function here
        for label in self.label_id.keys():
            if '-' not in label:
                self.full_list.append(label)
        return
    @staticmethod
    def count_samples_by_class(n, samples):
        """Count the number of samples for each class."""
        class_counts = np.zeros(n)  # one extra is for samples without motif
        # Iterate over the samples
        for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
            class_counts += type_protein
        return class_counts
    def __len__(self):
        return len(self.full_list)
    def __getitem__(self, index):
        anchor_label = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_label])
        # pos = random_positive(anchor, self.id_label, self.label_id)
        # neg = mine_negative(anchor, self.id_label, self.label_id, self.mine_neg)
        # a = torch.load('./data/esm_data/' + anchor + '.pt')
        # p = torch.load('./data/esm_data/' + pos + '.pt')
        # n = torch.load('./data/esm_data/' + neg + '.pt')
        # return format_esm(a), format_esm(p), format_esm(n)
        return

def custom_collate(batch):
    id, id_frags, fragments, target_frags, type_protein, sample_weight = zip(*batch)
    return id, id_frags, fragments, target_frags, type_protein, sample_weight


def prot_id_to_seq(seq_file):
    id2seq = {}
    with open(seq_file) as file:
        for line in file:
            id = line.strip().split("\t")[0]
            seq = line.strip().split("\t")[2]
            id2seq[id] = seq
    return id2seq


def split_protein_sequence(prot_id, sequence, targets, configs):
    fragment_length = configs.encoder.max_len - 2
    overlap = configs.encoder.frag_overlap
    fragments = []
    target_frags = []
    id_frags = []
    sequence_length = len(sequence)
    start = 0
    ind = 0

    while start < sequence_length:
        end = start + fragment_length
        if end > sequence_length:
            end = sequence_length
        fragment = sequence[start:end]
        target_frag = targets[:, start:end]
        if target_frag.shape[1] < fragment_length:
            pad = np.zeros([targets.shape[0], fragment_length-target_frag.shape[1]])
            target_frag = np.concatenate((target_frag, pad), axis=1)
        target_frags.append(target_frag)
        fragments.append(fragment)
        id_frags.append(prot_id+"@"+str(ind))
        ind += 1
        if start + fragment_length > sequence_length:
            break
        start += fragment_length - overlap

    return id_frags, fragments, target_frags


def fix_sample(motif_left, motif_right, label, label2idx, type_protein, targets):
    if motif_left == "None":
        motif_left = 0
    else:
        motif_left = int(motif_left)-1
    motif_right = int(motif_right)
    if label == "Thylakoid" and motif_left != 0:
        index_row = label2idx["chloroplast"]
        type_protein[index_row] = 1
        targets[index_row, motif_left-1] = 1
    return motif_left, motif_right, type_protein, targets


def prepare_samples(csv_file, configs):
    label2idx = {"Nucleus": 0, "ER": 1, "Peroxisome": 2, "Mitochondrion": 3, "Nucleus_export": 4,
                 "SIGNAL": 5, "chloroplast": 6, "Thylakoid": 7}
    samples = []
    n = configs.encoder.num_classes
    df = pd.read_csv(csv_file)
    row, col = df.shape
    for i in range(row):
        prot_id = df.loc[i, "Entry"]
        seq = df.loc[i, "Sequence"]
        targets = np.zeros([n, len(seq)])
        type_protein = np.zeros(n)
        motifs = df.loc[i, "MOTIF"].split("|")
        for motif in motifs:
            if not pd.isnull(motif):
                label = motif.split(":")[1]
                motif_left = motif.split(":")[0].split("-")[0]
                motif_right = motif.split(":")[0].split("-")[1]
                
                motif_left, motif_right, type_protein, targets = fix_sample(motif_left, motif_right, label, label2idx, type_protein, targets)
                if label in label2idx:
                    index_row = label2idx[label]
                    type_protein[index_row] = 1
                    if label in ["SIGNAL", "chloroplast", "Thylakoid", "Mitochondrion"]:
                        targets[index_row, motif_right-1] = 1
                    elif label == "Peroxisome" and motif_left == 0:
                        targets[index_row, motif_right-1] = 1
                    elif label == "Peroxisome" and motif_left != 0:
                        targets[index_row, motif_left] = 1
                    elif label == "ER":
                        targets[index_row, motif_left] = 1
                    elif label == "Nucleus" or label == "Nucleus_export":
                        targets[index_row, motif_left:motif_right] = 1
        id_frag_list, seq_frag_list, target_frag_list = split_protein_sequence(prot_id, seq, targets, configs)
        samples.append((prot_id, id_frag_list, seq_frag_list, target_frag_list, type_protein))

    return samples


def prepare_dataloaders(configs, valid_batch_number, test_batch_number):
    # id_to_seq = prot_id_to_seq(seq_file)
    if configs.train_settings.dataset == 'v2':
        samples = prepare_samples("./parsed_EC7_v2/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v2/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v2/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v2/split/type/partition.csv")
    elif configs.train_settings.dataset == 'v3':
        samples = prepare_samples("./parsed_EC7_v3/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v3/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v3/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v3/split/type/partition.csv")

    train_id = []
    val_id = []
    test_id = []
    id = cv.loc[:, 'entry']

    partition = cv.loc[:, 'partition']
    for i in range(len(id)):

        p = partition[i]
        d = id[i]
        if p == valid_batch_number:
            val_id.append(d)
        elif p == test_batch_number:
            test_id.append(d)
        else:
            train_id.append(d)

    train_sample = []
    valid_sample = []
    test_sample = []

    for i in samples:
        id = i[0]
        if id in train_id:
            train_sample.append(i)
        elif id in val_id:
            valid_sample.append(i)
        elif id in test_id:
            test_sample.append(i)

    random.seed(configs.fix_seed)
    # Shuffle the list
    random.shuffle(samples)

    # 改 local 参考Triplet_dataset_with_mine_EC
    train_dataset = LocalizationDataset(train_sample, configs=configs)
    valid_dataset = LocalizationDataset(valid_sample, configs=configs)
    test_dataset = LocalizationDataset(test_sample, configs=configs)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size, shuffle=True, collate_fn=custom_collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.valid_settings.batch_size, shuffle=True, collate_fn=custom_collate)

    # LocalizationDataset -> TripletDataset
    return {'train': train_dataloader, 'test': test_dataloader, 'valid': valid_dataloader}

if __name__ == '__main__':
    config_path = './config.yaml'
    with open(config_path) as file:
        configs_dict = yaml.full_load(file)

    configs_file = load_configs(configs_dict)

    dataloaders_dict = prepare_dataloaders(configs_file, 0, 1)

    for batch in dataloaders_dict['train']:
        (prot_id, id_frag_list, seq_frag_list, target_frag_nplist, type_protein_pt, sample_weight) = batch
        print("==========================")
        print(type(prot_id))
        print(prot_id)
        print(type(id_frag_list))
        print(id_frag_list)
        print(type(seq_frag_list))
        print(seq_frag_list)
        print(type(target_frag_nplist))
        print(target_frag_nplist)
        print(type(type_protein_pt))
        print(type_protein_pt)
        print(type(sample_weight))
        print(sample_weight)
        break

    print('done')
