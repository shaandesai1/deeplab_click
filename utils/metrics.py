import numpy as np
import torch

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
        masks1, masks2: [Height, Width, instances]
        """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    
    return overlaps

def get_bin_map(true_msk,pred_msk):
    h = 513
    w = 513
    
    ids = list(set(np.unique(true_msk)) -set([99]))
    all_collate = np.zeros((w,h,len(ids),2))
    #     for i,val in enumerate(ids):
    #         tsmsk = (true_msk == val)*1
    #         pmsk = (pred_msk == val)*1
    #         all_collate[:,:,i,0] = tsmsk
    #         all_collate[:,:,i,1] = pmsk
    all_collate[:,:,:,0] = true_msk.unsqueeze(2).expand(w,h,len(ids)).float() == (torch.ones((w,h,len(ids)))*torch.Tensor(ids)).float()
    all_collate[:,:,:,1] = pred_msk.unsqueeze(2).expand(w,h,len(ids)).float() == (torch.ones((w,h,len(ids)))*torch.Tensor(ids)).float()
    
    
    return compute_overlaps_masks(all_collate[:,:,:,1],all_collate[:,:,:,0])
