import math
import numpy as np
import torch
from matplotlib import pyplot as plt
import anndata as ad
import scanpy as sc
from data.base_dataset import BaseDataset
import os
import pandas as pd
import spateo as st
from scipy.sparse import lil_matrix, csr_matrix, vstack

class SCSMouseBrainDataset(BaseDataset):
    def __init__(self, opt, manager):
        BaseDataset.__init__(self, opt, manager)
        self.data_dir = os.path.join(opt.data_dir, opt.dataset.loc)
        self.tsv = os.path.join(self.data_dir, "Mouse_brain_Adult_GEM_bin1.tsv")
        self.tif = os.path.join(self.data_dir, "Mouse_brain_Adult.tif")
        self.adata = st.io.read_bgi_agg(self.tsv, self.tif)

        # Patch dir
        self.patch_dir = os.path.join(self.manager.get_run_dir(), "patch")
        if not os.path.exists(self.patch_dir):
            os.makedirs(self.patch_dir)
        self._get_info()

    def _get_info(self):
        r_all = []
        c_all = []
        with open(self.tsv) as fr:
            header = fr.readline()
            for line in fr:
                _, r, c, _ = line.split()
                r_all.append(int(r))
                c_all.append(int(c))
        self.rmax = np.max(r_all) - np.min(r_all)
        self.cmax = np.max(c_all) - np.min(c_all)
        self.n_patches = math.ceil(self.rmax / self.opt.dataset.patch_size) * math.ceil(self.cmax / self.opt.dataset.patch_size)
        self.logger.info(f'======> {self.n_patches} patches will be processed.')

        self.patch_index = list()
        for startr in range(0, self.rmax, self.opt.dataset.patch_size):
            for startc in range(0, self.cmax, self.opt.dataset.patch_size):
                self.patch_index.append([startr, startc])
        self._set_patch_dataset()

    def __len__(self):
        return len(self.patch_dataset_list)

    def __getitem__(self, index):
        return self.patch_dataset_list[index]

    def _set_patch_dataset(self):
        self.patch_dataset_list = list()
        for i in self.patch_index:
            try:
                x_train, x_train_pos, y_train, y_binary_train,  x_test, x_test_pos = self._preprocess_one_patch(
                    i[0], i[1], self.opt.dataset.patch_size, self.opt.dataset.bin_size, self.opt.dataset.n_neighbor)
                temp_d = {
                    "loc": i,
                    "patch_size": self.opt.dataset.patch_size,
                    "x_train": x_train,
                    "x_train_pos": x_train_pos,
                    "y_train": y_train,
                    "y_binary_train": y_binary_train,
                    "x_test": x_test,
                    "x_test_pos": x_test_pos
                }
                self.patch_dataset_list.append(PerPatchDataset(temp_d, self.opt, self.manager))
            except Exception as e:
                print(e)
                self.logger.info('======> Patch ' + str(i[0]) + ':' + str(
                    i[1]) + ' failed. This could be due to no nuclei detected by Watershed or too few RNAs in the patch.')
            if self.opt.dataset.max_patch_num > 0:
                if len(self.patch_dataset_list) == self.opt.dataset.max_patch_num:
                    break

    def _preprocess_one_patch(self, startx, starty, patch_size, bin_size, n_neighbor, align='rigid'):
        adatasub = self.adata[int(startx):int(startx)+int(patch_size),int(starty):int(starty)+int(patch_size)]

        startx = str(startx)
        starty = str(starty)

        adatasub.layers['unspliced'] = adatasub.X
        patch_size_x = adatasub.X.shape[0]
        patch_size_y = adatasub.X.shape[1]

        # align staining image with bins
        if align:
            st.cs.refine_alignment(adatasub, mode = align, transform_layers=['stain'])

        st.cs.mask_nuclei_from_stain(adatasub, otsu_classes = 4, otsu_index=1)
        st.cs.find_peaks_from_mask(adatasub, 'stain', 7)
        st.cs.watershed(adatasub, 'stain', 5, out_layer='watershed_labels')
        # adatasub.write(os.path.join(self.patch_dir, startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.h5ad'))

        watershed2x = {}
        watershed2y = {}
        for i in range(adatasub.layers['watershed_labels'].shape[0]):
            for j in range(adatasub.layers['watershed_labels'].shape[1]):
                if adatasub.layers['watershed_labels'][i, j] == 0:
                    # Background
                    continue
                if adatasub.layers['watershed_labels'][i, j] in watershed2x:
                    watershed2x[adatasub.layers['watershed_labels'][i, j]].append(i)
                    watershed2y[adatasub.layers['watershed_labels'][i, j]].append(j)
                else:
                    watershed2x[adatasub.layers['watershed_labels'][i, j]] = [i]
                    watershed2y[adatasub.layers['watershed_labels'][i, j]] = [j]

        watershed2center = {}
        sizes = []

        for nucleus in watershed2x:
            watershed2center[nucleus] = [np.mean(watershed2x[nucleus]), np.mean(watershed2y[nucleus])]
            sizes.append(len(watershed2x[nucleus]))

        xall = []
        yall = []
        with open(self.tsv) as fr:
            header = fr.readline()
            for line in fr:
                gene, x, y, count = line.split()
                xall.append(int(x))
                yall.append(int(y))
        xmin = np.min(xall)
        ymin = np.min(yall)

        # Find all the genes in the range
        geneid = {}
        genecnt = 0
        id2gene = {}
        with open(self.tsv) as fr:
            header = fr.readline()
            for line in fr:
                gene, x, y, count = line.split()
                if gene not in geneid:
                    geneid[gene] = genecnt
                    id2gene[genecnt] = gene
                    genecnt += 1

        idx2exp = {}
        downrs = bin_size

        with open(self.tsv) as fr:
            header = fr.readline()
            for line in fr:
                gene, x, y, count = line.split()
                x = int(x) - xmin
                y = int(y) - ymin
                if gene not in geneid:
                    continue
                if int(x) < int(startx) or int(x) >= int(startx) + int(patch_size_x) or int(y) < int(starty) or int(
                        y) >= int(starty) + int(patch_size_y):
                    continue
                idx = int(math.floor((int(x) - int(startx)) / downrs) * math.ceil(patch_size_y / downrs) + math.floor(
                    (int(y) - int(starty)) / downrs))
                if idx not in idx2exp:
                    idx2exp[idx] = {}
                    idx2exp[idx][geneid[gene]] = int(count)
                elif geneid[gene] not in idx2exp[idx]:
                    idx2exp[idx][geneid[gene]] = int(count)
                else:
                    idx2exp[idx][geneid[gene]] += int(count)

        all_exp_merged_bins = lil_matrix(
            (int(math.ceil(patch_size_x / downrs) * math.ceil(patch_size_y / downrs)), genecnt), dtype=np.int8)
        for idx in idx2exp:
            for gid in idx2exp[idx]:
                all_exp_merged_bins[idx, gid] = idx2exp[idx][gid]
                # print(idx, gid, idx2exp[idx][gid])
        all_exp_merged_bins = all_exp_merged_bins.tocsr()

        # Shape: [patch_size * patch_size / (downrs * downrs), Gene_num]
        all_exp_merged_bins_ad = ad.AnnData(
            all_exp_merged_bins,
            obs=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[0])]),
            var=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[1])]),
        )
        sc.pp.highly_variable_genes(all_exp_merged_bins_ad, n_top_genes=2000, flavor='seurat_v3', span=1.0)
        selected_index = all_exp_merged_bins_ad.var[all_exp_merged_bins_ad.var.highly_variable].index
        selected_index = list(selected_index)
        selected_index = [int(i) for i in selected_index]

        # Save
        self.selected_index = selected_index

        # Shape: [patch_size * patch_size / (downrs * downrs), Selected_Gene_num]
        all_exp_merged_bins = all_exp_merged_bins.toarray()[:, selected_index]

        offsets = []
        for dis in range(1, 11):
            for dy in range(-dis, dis + 1):
                offsets.append([-dis * downrs, dy * downrs])
            for dy in range(-dis, dis + 1):
                offsets.append([dis * downrs, dy * downrs])
            for dx in range(-dis + 1, dis):
                offsets.append([dx * downrs, -dis * downrs])
            for dx in range(-dis + 1, dis):
                offsets.append([dx * downrs, dis * downrs])

        # Save mid result
        x_train_tmp = []
        # X_train: for every exist transcription i j: [[n_neighbor, Gene_num],[n_neighbor, Gene_num],...]
        x_train = []
        # x_train_pos: for every exist transcription i j: [[n_neighbor, Gene_num],[n_neighbor, Gene_num],...]
        x_train_pos = []
        # y_train: label of the n_neighbor spots
        y_train = []
        # 1 for cell / -1 for bg
        y_binary_train = []

        x_train_bg_tmp = []
        x_train_bg = []
        x_train_pos_bg = []
        y_train_bg = []
        y_binary_train_bg = []

        # Other test sample where transcription exists
        x_test_tmp = []
        x_test = []
        x_test_pos = []

        for i in range(adatasub.layers['watershed_labels'].shape[0]):
            for j in range(adatasub.layers['watershed_labels'].shape[1]):
                if (not i % downrs == 0) or (not j % downrs == 0):
                    continue
                idx = int(math.floor(i / downrs) * math.ceil(patch_size_y / downrs) + math.floor(j / downrs))
                if adatasub.layers['watershed_labels'][i, j] > 0:
                    if idx >= 0 and idx < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx, :]) > 0:
                        x_train_sample = [all_exp_merged_bins[idx, :]]
                        x_train_pos_sample = [[i, j]]
                        y_train_sample = [watershed2center[adatasub.layers['watershed_labels'][i, j]]]
                        for dx, dy in offsets:
                            if len(x_train_sample) == n_neighbor:
                                break
                            x = i + dx
                            y = j + dy
                            if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= \
                                    adatasub.layers['watershed_labels'].shape[1]:
                                continue
                            idx_nb = int(
                                math.floor(x / downrs) * math.ceil(patch_size_y / downrs) + math.floor(y / downrs))
                            if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(
                                    all_exp_merged_bins[idx_nb, :]) > 0:
                                x_train_sample.append(all_exp_merged_bins[idx_nb, :])
                                x_train_pos_sample.append([x, y])
                        if len(x_train_sample) < n_neighbor:
                            continue
                        x_train_tmp.append(x_train_sample)
                        if len(x_train_tmp) > 500:
                            x_train.extend(x_train_tmp)
                            x_train_tmp = []
                        x_train_pos.append(x_train_pos_sample)
                        y_train.append(y_train_sample)
                        y_binary_train.append(1)
                else:
                    if idx >= 0 and idx < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx, :]) > 0:
                        backgroud = True
                        for nucleus in watershed2center:
                            # If closed to a nuleus, then background = False
                            if (i - watershed2center[nucleus][0]) ** 2 + (
                                    j - watershed2center[nucleus][1]) ** 2 <= 900 or adatasub.layers['stain'][
                                i, j] > 10:
                                backgroud = False
                                break
                        if backgroud:
                            # If to much bg, then continue
                            if len(x_train_bg) + len(x_train_bg_tmp) >= len(x_train) + len(x_train_tmp):
                                continue
                            x_train_sample = [all_exp_merged_bins[idx, :]]
                            x_train_pos_sample = [[i, j]]
                            y_train_sample = [[-1, -1]]
                            for dx, dy in offsets:
                                if len(x_train_sample) == n_neighbor:
                                    break
                                x = i + dx
                                y = j + dy
                                if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= \
                                        adatasub.layers['watershed_labels'].shape[1]:
                                    continue
                                idx_nb = int(
                                    math.floor(x / downrs) * math.ceil(patch_size_y / downrs) + math.floor(y / downrs))
                                if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(
                                        all_exp_merged_bins[idx_nb, :]) > 0:
                                    x_train_sample.append(all_exp_merged_bins[idx_nb, :])
                                    x_train_pos_sample.append([x, y])
                            if len(x_train_sample) < n_neighbor:
                                continue
                            x_train_bg_tmp.append(x_train_sample)
                            if len(x_train_bg_tmp) > 500:
                                x_train_bg.extend(x_train_bg_tmp)
                                x_train_bg_tmp = []
                            # print(len(x_train_bg_tmp))
                            x_train_pos_bg.append(x_train_pos_sample)
                            y_train_bg.append(y_train_sample)
                            y_binary_train_bg.append(0)
                        else:
                            x_test_sample = [all_exp_merged_bins[idx, :]]
                            x_test_pos_sample = [[i, j]]
                            for dx, dy in offsets:
                                if len(x_test_sample) == n_neighbor:
                                    break
                                x = i + dx
                                y = j + dy
                                if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= \
                                        adatasub.layers['watershed_labels'].shape[1]:
                                    continue
                                idx_nb = int(
                                    math.floor(x / downrs) * math.ceil(patch_size_y / downrs) + math.floor(y / downrs))
                                if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(
                                        all_exp_merged_bins[idx_nb, :]) > 0:
                                    x_test_sample.append(all_exp_merged_bins[idx_nb, :])
                                    x_test_pos_sample.append([x, y])
                            if len(x_test_sample) < n_neighbor:
                                continue
                            x_test_tmp.append(x_test_sample)
                            if len(x_test_tmp) > 500:
                                x_test.extend(x_test_tmp)
                                x_test_tmp = []
                            x_test_pos.append(x_test_pos_sample)  #

        x_train.extend(x_train_tmp)
        x_train_bg.extend(x_train_bg_tmp)
        x_test.extend(x_test_tmp)

        x_train = np.array(x_train)
        x_train_pos = np.array(x_train_pos)
        y_train = np.vstack(y_train)
        y_binary_train = np.array(y_binary_train)
        x_train_bg = np.array(x_train_bg)
        x_train_pos_bg = np.array(x_train_pos_bg)
        y_train_bg = np.vstack(y_train_bg)
        y_binary_train_bg = np.array(y_binary_train_bg)

        bg_index = np.arange(len(x_train_bg))
        np.random.shuffle(bg_index)
        x_train = np.vstack((x_train, x_train_bg[bg_index[:len(x_train)]]))
        x_train_pos = np.vstack((x_train_pos, x_train_pos_bg[bg_index[:len(x_train_pos)]]))
        y_train = np.vstack((y_train, y_train_bg[bg_index[:len(y_train)]]))
        y_binary_train = np.hstack((y_binary_train, y_binary_train_bg[bg_index[:len(y_binary_train)]]))

        x_test = np.array(x_test)
        x_test_pos = np.array(x_test_pos)
        return x_train, x_train_pos, y_train, y_binary_train, x_test, x_test_pos

class PerPatchDataset(BaseDataset):
    def __init__(self, processed_patch_dict, opt, manager):
        BaseDataset.__init__(self, opt, manager)
        # dict = {
        #     "loc": i,
        #     "patch_size": self.opt.dataset.patch_size,
        #     "x_train": x_train,
        #     "x_train_pos": x_train_pos,
        #     "y_train": y_train,
        #     "y_binary_train": y_binary_train,
        #     "x_test": x_test,
        #     "x_test_pos": x_test_pos
        # }
        self.patch_dict = processed_patch_dict
        self.mode = 'train'

        self.patch_dict["x_train"] = torch.from_numpy(self.patch_dict["x_train"])
        self.patch_dict["x_train_pos"] = torch.from_numpy(self.patch_dict["x_train_pos"])
        self.patch_dict["y_train"] = torch.from_numpy(self.patch_dict["y_train"])
        self.patch_dict["y_binary_train"] = torch.from_numpy(self.patch_dict["y_binary_train"])
        self.patch_dict["x_test"] = torch.from_numpy(self.patch_dict["x_test"])
        self.patch_dict["x_test_pos"] = torch.from_numpy(self.patch_dict["x_test_pos"])

    def set_mode(self, mode):
        # Must be setup after init
        assert mode in ['train', 'test'],"Value of mode must be train or test"
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return self.patch_dict["x_train"].shape[0]
        else:
            return self.patch_dict["x_test"].shape[0]

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.patch_dict["x_train"][index], self.patch_dict["x_train_pos"][index], self.patch_dict["y_train"][index], self.patch_dict["y_binary_train"][index]
        else:
            return self.patch_dict["x_test"][index], self.patch_dict["x_test_pos"][index]