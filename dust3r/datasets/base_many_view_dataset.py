from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset



class BaseManyViewDataset(BaseStereoViewDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_ratio = None

    def sample_frames(self, img_idxs, rng, no_interval=False):
        num_frames = self.num_frames
        if self.train_ratio is not None:
            thresh = int(self.min_thresh + self.train_ratio * (self.max_thresh - self.min_thresh))
        else:
            thresh = 8
        img_indices = list(range(len(img_idxs)))
        
        selected_indices = []
        
        initial_valid_range = max(len(img_indices)//num_frames, len(img_indices) - thresh * (num_frames - 1))
        current_index = rng.choice(img_indices[:initial_valid_range])
        if no_interval:
            return list(range(current_index, current_index+num_frames))
        selected_indices.append(current_index)
        
        while len(selected_indices) < num_frames:
            next_min_index = current_index + 1
            next_max_index = min(current_index + thresh, len(img_indices) - (num_frames - len(selected_indices)))
            # print("next min", next_min_index, next_max_index)
            possible_indices = [i for i in range(next_min_index, next_max_index + 1) if i not in selected_indices]
        
            if not possible_indices:
                break
            
            current_index = rng.choice(possible_indices)
            selected_indices.append(current_index)
        
        if len(selected_indices) < num_frames:
            return self.sample_frames(img_idxs, rng)

        selected_img_ids = [img_idxs[i] for i in selected_indices]
        
        if rng.choice([True, False]):
            selected_img_ids.reverse()
        
        return selected_img_ids
    

    def sample_frame_idx(self, img_idxs, rng, full_video=False, no_interval=False):
        if not full_video:
            img_idxs = self.sample_frames(img_idxs, rng, no_interval)
        else:
            img_idxs = img_idxs[::self.kf_every]
        
        return img_idxs


    