import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.feature.texture import graycomatrix

#### Spatial Cooccurrence Methods  ####

class BandMatrix:

    def __init__(self, band, channel):
        self.channel = channel
        self.channel_mat = self.build_mats(band).T.reshape(4,256,256)

    def build_mats(self, band):
        return graycomatrix(band, distances=[1], angles=[0, np.pi/2, np.pi/4, 5*(np.pi/4)], levels=256)
    
    '''
    def build_mats(self, band, len_col, len_row):
        """ after: https://arxiv.org/pdf/2007.10466.pdf
        """
        h_mat = np.zeros(256*256).reshape(256,256)
        v_mat = np.zeros(256*256).reshape(256,256)
        d_mat = np.zeros(256*256).reshape(256,256)
        ad_mat = np.zeros(256*256).reshape(256,256)
        for c, row in enumerate(band):
            for r, val in enumerate(row):
                if r < len_row-1:
                    h_val = band[c,r+1]
                if c < len_col-1:
                    v_val = band[c+1,r]
                if r < len_row-1 and c < len_col-1:
                    d_val = band[c+1,r+1]
                if c >=1 and r >= 1:
                    ad_val = band[c-1, r-1]
                    ad_mat[val, ad_val] += 1
                h_mat[val, h_val] += 1
                v_mat[val, v_val] += 1
                d_mat[val, d_val] += 1
        return h_mat,v_mat,d_mat, ad_mat
    '''
    
    def show_band_heatmap(self, typ):
        if typ == 'vertical':
            ax = sns.heatmap(self.v_mat)
            plt.show()
        elif typ == 'horizontal':
            ax = sns.heatmap(self.h_mat)
            plt.show()
        elif typ == 'diagonal':
            ax = sns.heatmap(self.d_mat)
            plt.show()
        elif typ == 'antidiag':
            ax = sns.heatmap(self.ad_mat)
            plt.show()
        else:
            assert ValueError('WRONG INPUT STRING FOR COOCURRENCE DIRECTION')


class BandCMatrix:

    def __init__(self, img):
        self.mat_r = BandMatrix(img.r_tensor, 'red')
        self.mat_g = BandMatrix(img.g_tensor, 'green')
        self.mat_b = BandMatrix(img.b_tensor, 'blue')
    
    def get_concat_matrix(self):
        return np.concatenate([self.mat_r.channel_mat, self.mat_g.channel_mat, self.mat_b.channel_mat], dtype=np.uint8)


class CrossBandMatrix:

    def __init__(self, combination, channel, offset_type):
        self.channel = channel
        self.mat = self.build_mats(combination, offset_type)

    def build_mats(self, combination, offset_type):
        """ after: https://arxiv.org/pdf/2007.12909.pdf
        """
        off_x, off_y = offset_type
        origin_mat, target_mat = combination
        combination_mat = np.zeros(256*256).reshape(256,256)
        for c in range(len(origin_mat)-off_x):
            for r in range(len(origin_mat[0])-off_y):
                origin_val = origin_mat[c,r]
                target_val = target_mat[c+off_x,r+off_y]
                combination_mat[origin_val, target_val] += 1
        return combination_mat

    def show_crossband_heatmap(self):
        ax = sns.heatmap(self.mat)
        plt.show()


class CrossCMatrix:

    def __init__(self, img, offset_type):
        self.mat_RG = CrossBandMatrix((img.r_tensor,img.g_tensor), 'red_green', offset_type)
        self.mat_RB = CrossBandMatrix((img.r_tensor,img.b_tensor), 'red_blue', offset_type)
        self.mat_GB = CrossBandMatrix((img.g_tensor,img.b_tensor), 'green_blue', offset_type)
    
    def get_concat_matrix(self):
        return np.stack([self.mat_RG.mat, self.mat_RB.mat, self.mat_GB.mat])

class QuickResidualMat:

    def __init__(self, img, offsets, offset_stride_d, residual_direction, trunc_threshold, normalization):
        self.len_col, self.len_row = img.len_col, img.len_row
        self.normalization = normalization
        self.residual_direction = residual_direction
        self.trunc_threshold = trunc_threshold

        # get HSV
        h, s , _ = img.img.convert('HSV').split()
        self.h_tensor = self.get_residual(np.array(h)) 
        self.s_tensor = self.get_residual(np.array(s))

        # get YCbCr
        _, cb , cr = img.img.convert('YCbCr').split()
        self.cb_tensor = self.get_residual(np.array(cb))
        self.cr_tensor = self.get_residual(np.array(cr))


        self.h_cooc_mat = self.build_mat(self.h_tensor, offsets, offset_stride_d)
        self.s_cooc_mat = self.build_mat(self.s_tensor, offsets, offset_stride_d)
        self.cb_cooc_mat = self.build_mat(self.cb_tensor, offsets, offset_stride_d)
        self.cr_cooc_mat = self.build_mat(self.cr_tensor, offsets, offset_stride_d)

        print('processed for channel combo')

    def get_residual(self, channel_mat):
        """ after: https://arxiv.org/pdf/1808.07276.pdf
            calculates differential residuals of image channels. Formula 3+4
        """
        if self.residual_direction == 'horizontal':
            zero_padding = np.zeros(self.len_col).reshape(1, self.len_col)
            channel_adj_mask = np.concatenate((channel_mat.T[1:], zero_padding), axis=0).T
            truncated_mat = np.clip((channel_mat - channel_adj_mask), a_min=-self.trunc_threshold, a_max=self.trunc_threshold).astype(int)
            return truncated_mat.reshape(1,self.len_col, self.len_row)
        elif self.residual_direction == 'vertical':
            zero_padding = np.zeros(self.len_row).reshape(1, self.len_row)
            channel_adj_mask = np.concatenate((channel_mat[1:], zero_padding), axis=0)
            truncated_mat = np.clip((channel_mat - channel_adj_mask), a_min=-self.trunc_threshold, a_max=self.trunc_threshold).astype(int)
            return truncated_mat.reshape(1,self.len_col, self.len_row)
        elif self.residual_direction == 'both':
            #horizontal
            zero_padding = np.zeros(self.len_col).reshape(1, self.len_col)
            channel_adj_mask = np.concatenate((channel_mat.T[1:], zero_padding), axis=0).T
            hori_residual = np.clip((channel_mat - channel_adj_mask), a_min=-self.trunc_threshold, a_max=self.trunc_threshold).astype(int)
            #vertical
            zero_padding = np.zeros(self.len_row).reshape(1, self.len_row)
            channel_adj_mask = np.concatenate((channel_mat[1:], zero_padding), axis=0)
            vert_residual = np.clip((channel_mat - channel_adj_mask), a_min=-self.trunc_threshold, a_max=self.trunc_threshold).astype(int)
            return np.stack((hori_residual, vert_residual))
        else:
            raise ValueError('WRONG INPUT FOR RESIDUAL DIRECTION. ENTER HORIZONTAL, VERTICAL OR BOTH')

    def build_sub_mat_horizontal(self, offset, band, offset_stride_d, pixel_range):
        """ after: https://arxiv.org/pdf/1808.07276.pdf
            calculates cooccurrence matrix. Formula 5
            gives matrix 2tau*2tau*d, where d is the offset stride and the pixel range -tau < p < tau
        """
        _, off_y = offset
        row_iter_length = self.len_row-off_y
        cooc_mat = np.zeros(pixel_range*pixel_range*(offset_stride_d+1)).reshape((offset_stride_d+1), pixel_range, pixel_range)
        lower_bound_by_offset = offset_stride_d+2
        for c in range(self.len_col):
            for r in range(row_iter_length):
                origin_val = band[c,r]
                lower_bound_by_dim = self.len_row-r
                lower_bound = min(lower_bound_by_dim,lower_bound_by_offset)
                for m in range(1,lower_bound):
                    target_val = band[c,r+m]
                    cooc_mat[m-1, origin_val, target_val] += 1
        return (1/self.normalization)*cooc_mat
     
    def build_sub_mat_vertical(self, offset, band, offset_stride_d, pixel_range):
        """ after: https://arxiv.org/pdf/1808.07276.pdf
            calculates cooccurrence matrix. Formula 5
            gives matrix 2tau*2tau*d, where d is the offset stride and the pixel range -tau < p < tau
        """
        off_x, _ = offset
        col_iter_length = self.len_col-off_x
        cooc_mat = np.zeros(pixel_range*pixel_range*(offset_stride_d+1)).reshape((offset_stride_d+1), pixel_range, pixel_range)
        lower_bound_by_offset = offset_stride_d+2
        for c in range(col_iter_length):
            lower_bound_by_dim = self.len_col-c
            lower_bound = min(lower_bound_by_dim,lower_bound_by_offset)
            for r in range(self.len_row):
                origin_val = band[c,r]
                for m in range(1,lower_bound):
                    target_val = band[c+m,r]
                    cooc_mat[m-1, origin_val, target_val] += 1
        return (1/self.normalization)*cooc_mat
    
    def build_mat(self, bands, offsets, offset_stride_d):
        pixel_range = 2*self.trunc_threshold+1
        cooc_mat = np.zeros(pixel_range*pixel_range*(offset_stride_d+1)).reshape((offset_stride_d+1), pixel_range, pixel_range)
        if len(offsets) > 1:
            off_horizontal, off_vertical = offsets
            for band in bands:
                cooc_mat += self.build_sub_mat_horizontal(off_horizontal, band, offset_stride_d, pixel_range)
                cooc_mat += self.build_sub_mat_vertical(off_vertical, band, offset_stride_d, pixel_range)
        else:
            offset = offsets[0]
            if offset[0] == 1:
                for band in bands:
                    cooc_mat += self.build_sub_mat_vertical(offset, band, offset_stride_d, pixel_range)
            else:
                for band in bands:
                    cooc_mat += self.build_sub_mat_horizontal(offset, band, offset_stride_d, pixel_range)
        return self.permutation_for_negative_cooccurrence_indexes(cooc_mat)
    
    def permutation_for_negative_cooccurrence_indexes(self, array):
        dim = 2*self.trunc_threshold+1
        permutation = [i for i in range(int(dim/2)+1, dim)]
        permutation.extend([i for i in range(int(dim/2)+1)])
        permuted = array[:, :, permutation][:, permutation, :]
        return permuted


class ConvImg:

    def __init__(self, img, offsets, offset_stride_d, conv_type, residual_processing, residual_direction, trunc_threshold, normalization):
        self.len_col, self.len_row = img.len_col, img.len_row
        self.channel_type = conv_type
        self.normalization = normalization
        self.residual_direction = residual_direction
        if residual_processing:
            self.trunc_threshold = trunc_threshold
            if conv_type == 'RGB':
                self.c1, self.c2, self.c3 = 'r', 'g', 'b'
                c1_tensor, c2_tensor, c3_tensor = img.get_channels_tensor()
                self.c1_tensor, self.c2_tensor, self.c3_tensor = self.get_residual(c1_tensor), self.get_residual(c2_tensor), self.get_residual(c3_tensor)
            elif conv_type == 'HSV':
                self.c1, self.c2, self.c3 = 'h', 's', 'v'
                c1, c2 , c3 = img.img.convert(conv_type).split()
                self.c1_tensor = self.get_residual(np.array(c1)) 
                self.c2_tensor = self.get_residual(np.array(c2))
                self.c3_tensor = self.get_residual(np.array(c3))
            else:
                self.c1, self.c2, self.c3 = 'y', 'cb', 'cr'
                c1, c2 , c3 = img.img.convert(conv_type).split()
                self.c1_tensor = self.get_residual(np.array(c1)) 
                self.c2_tensor = self.get_residual(np.array(c2))
                self.c3_tensor = self.get_residual(np.array(c3))
        else:
            if conv_type == 'RGB':
                self.c1, self.c2, self.c3 = 'r', 'g', 'b'
                self.c1_tensor, self.c2_tensor, self.c3_tensor = img.get_channels_tensor()
            elif conv_type == 'HSV':
                self.c1, self.c2, self.c3 = 'h', 's', 'v'
                c1, c2 , c3 = img.img.convert(conv_type).split()
                self.c1_tensor = np.array(c1) 
                self.c2_tensor = np.array(c2)
                self.c3_tensor = np.array(c3)
            else:
                self.c1, self.c2, self.c3 = 'y', 'cb', 'cr'
                c1, c2 , c3 = img.img.convert(conv_type).split()
                self.c1_tensor = np.array(c1) 
                self.c2_tensor = np.array(c2)
                self.c3_tensor = np.array(c3)

        #self.correlation_dict = self.corr_of_adjacent_pixels()
        #self.cooc_mat = self.build_mat(self.c1_tensor, offsets, offset_stride_d)
    
    def get_channels(self):
        return ((self.c1, self.c1_tensor),(self.c2, self.c2_tensor),(self.c3, self.c3_tensor))

    def corr_of_adjacent_pixels(self):
        """The larger the r_c, the higher correlation between the
           adjacent pixel values in I_c
        """
        return {ch_name: self.corr_adj_pix_channel(channel) for ch_name, channel in self.get_channels()}
    
    def corr_adj_pix_channel(self, channel_mat):
        """ after: https://arxiv.org/pdf/1808.07276.pdf
            calculates correlation of adjacent pixels of given channels. Formula 1
        """
        zero_padding = np.zeros(self.len_col).reshape(1,self.len_col) 
        channel_mat_std = channel_mat - channel_mat.mean()
        channel_adj_mask = np.concatenate((channel_mat_std.T[1:],zero_padding),axis=0).T
        num = np.multiply(channel_mat_std,channel_adj_mask).sum()
        frob_std = np.linalg.norm(channel_mat_std, 'fro')
        frob_mask = np.linalg.norm(channel_adj_mask, 'fro')
        denom = frob_std*frob_mask
        return num/denom

    def get_residual(self, channel_mat):
        """ after: https://arxiv.org/pdf/1808.07276.pdf
            calculates differential residuals of image channels. Formula 3+4
        """
        if self.residual_direction == 'horizontal':
            zero_padding = np.zeros(self.len_col).reshape(1, self.len_col)
            channel_adj_mask = np.concatenate((channel_mat.T[1:], zero_padding), axis=0).T
            truncated_mat = np.clip((channel_mat - channel_adj_mask), a_min=-self.trunc_threshold, a_max=self.trunc_threshold).astype(int)
            return truncated_mat.reshape(1,self.len_col, self.len_row)
        elif self.residual_direction == 'vertical':
            zero_padding = np.zeros(self.len_row).reshape(1, self.len_row)
            channel_adj_mask = np.concatenate((channel_mat[1:], zero_padding), axis=0)
            truncated_mat = np.clip((channel_mat - channel_adj_mask), a_min=-self.trunc_threshold, a_max=self.trunc_threshold).astype(int)
            return truncated_mat.reshape(1,self.len_col, self.len_row)
        elif self.residual_direction == 'both':
            #horizontal
            zero_padding = np.zeros(self.len_col).reshape(1, self.len_col)
            channel_adj_mask = np.concatenate((channel_mat.T[1:], zero_padding), axis=0).T
            hori_residual = np.clip((channel_mat - channel_adj_mask), a_min=-self.trunc_threshold, a_max=self.trunc_threshold).astype(int)
            #vertical
            zero_padding = np.zeros(self.len_row).reshape(1, self.len_row)
            channel_adj_mask = np.concatenate((channel_mat[1:], zero_padding), axis=0)
            vert_residual = np.clip((channel_mat - channel_adj_mask), a_min=-self.trunc_threshold, a_max=self.trunc_threshold).astype(int)
            return np.stack((hori_residual, vert_residual))
        else:
            raise ValueError('WRONG INPUT FOR RESIDUAL DIRECTION. ENTER HORIZONTAL, VERTICAL OR BOTH')

    def build_sub_mat_horizontal(self, offset, band, offset_stride_d, pixel_range):
        """ after: https://arxiv.org/pdf/1808.07276.pdf
            calculates cooccurrence matrix. Formula 5
            gives matrix 2tau*2tau*d, where d is the offset stride and the pixel range -tau < p < tau
        """
        _, off_y = offset
        row_iter_length = self.len_row-off_y
        cooc_mat = np.zeros(pixel_range*pixel_range*(offset_stride_d+1)).reshape((offset_stride_d+1), pixel_range, pixel_range)
        lower_bound_by_offset = offset_stride_d+2
        for c in range(self.len_col):
            for r in range(row_iter_length):
                origin_val = band[c,r]
                lower_bound_by_dim = self.len_row-r
                lower_bound = min(lower_bound_by_dim,lower_bound_by_offset)
                for m in range(1,lower_bound):
                    target_val = band[c,r+m]
                    cooc_mat[m-1, origin_val, target_val] += 1
        return (1/self.normalization)*cooc_mat

    def build_sub_mat_vertical(self, offset, band, offset_stride_d, pixel_range):
        """ after: https://arxiv.org/pdf/1808.07276.pdf
            calculates cooccurrence matrix. Formula 5
            gives matrix 2tau*2tau*d, where d is the offset stride and the pixel range -tau < p < tau
        """
        off_x, _ = offset
        col_iter_length = self.len_col-off_x
        cooc_mat = np.zeros(pixel_range*pixel_range*(offset_stride_d+1)).reshape((offset_stride_d+1), pixel_range, pixel_range)
        lower_bound_by_offset = offset_stride_d+2
        for c in range(col_iter_length):
            lower_bound_by_dim = self.len_col-c
            lower_bound = min(lower_bound_by_dim,lower_bound_by_offset)
            for r in range(self.len_row):
                origin_val = band[c,r]
                for m in range(1,lower_bound):
                    target_val = band[c+m,r]
                    cooc_mat[m-1, origin_val, target_val] += 1
        return (1/self.normalization)*cooc_mat
    
    def build_mat(self, bands, offsets, offset_stride_d):
        pixel_range = 2*self.trunc_threshold+1
        cooc_mat = np.zeros(pixel_range*pixel_range*(offset_stride_d+1)).reshape((offset_stride_d+1), pixel_range, pixel_range)
        if len(offsets) > 1:
            off_horizontal, off_vertical = offsets
            for band in bands:
                cooc_mat += self.build_sub_mat_horizontal(off_horizontal, band, offset_stride_d, pixel_range)
                cooc_mat += self.build_sub_mat_vertical(off_vertical, band, offset_stride_d, pixel_range)
        else:
            offset = offsets[0]
            if offset[0] == 1:
                for band in bands:
                    cooc_mat += self.build_sub_mat_vertical(offset, band, offset_stride_d, pixel_range)
            else:
                for band in bands:
                    cooc_mat += self.build_sub_mat_horizontal(offset, band, offset_stride_d, pixel_range)
        return self.permutation_for_negative_cooccurrence_indexes(cooc_mat)
    
    def permutation_for_negative_cooccurrence_indexes(self, array):
        dim = 2*self.trunc_threshold+1
        permutation = [i for i in range(int(dim/2)+1, dim)]
        permutation.extend([i for i in range(int(dim/2)+1)])
        permuted = array[:, :, permutation][:, permutation, :]
        return permuted
    

class ResidualCMatrix:

    def __init__(self, img, offsets, offset_stride_d, residual_processing, residual_direction, trunc_threshold, normalization):
        self.r_g_b = ConvImg(img, offsets, offset_stride_d, 'RGB', residual_processing, residual_direction, trunc_threshold, normalization)
        self.h_s_v = ConvImg(img, offsets, offset_stride_d, 'HSV', residual_processing, residual_direction, trunc_threshold, normalization)
        self.y_cb_cr = ConvImg(img, offsets, offset_stride_d, 'YCbCr', residual_processing, residual_direction, trunc_threshold, normalization)
        '''
        self.correlation_dict = {**self.r_g_b.correlation_dict, **self.h_s_v.correlation_dict, **self.y_cb_cr.correlation_dict}
        self.correlation_hist = np.array(self.correlation_dict.values())
        '''

    def show_histogram(self):
        ind = np.arange(len(self.correlation_dict))
        plt.bar(ind, list(self.correlation_dict.values()))
        plt.xticks(ind, list(self.correlation_dict.keys()))
        plt.show()
        

#### ---------- ####

class SpatialProcessor:

    def __init__(self, img, offset, residual_processing, residual_direction, trunc_threshold, normalization, cross_offset=None):
        self.img = img
        self.ca = ResidualCMatrix(img, offset, residual_processing, residual_direction, trunc_threshold, normalization)
        #self.cooc_mat = CoocMatrix(img)
        #self.cross_cooc_mat = CrossCoocMatrix(img, cross_offset)

                

if __name__ == '__main__':
    pass