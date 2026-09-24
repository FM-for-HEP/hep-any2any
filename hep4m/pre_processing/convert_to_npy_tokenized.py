import logging
import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

def convert(modality, filelist, output_file_prefix, sort_by_var=None, ev_num_fp=None, repeat_pos_tokens=False):
        
    # 1. determine number of codebooks, and total elements
    for fp_i, fp in tqdm(enumerate(filelist), desc="\n   Determining Schema", total=len(filelist)):
        f = uproot.open(fp)
        tree = f['event_tree']

        if fp_i == 0:
            n_codebooks = len([x for x in tree.keys() \
                if (f'{modality}_token' in x and f'n{modality}_token' not in x)])
            tokens_vname = [f'{modality}_token_{i}' for i in range(n_codebooks)]

            n_pos_codebooks = len([x for x in tree.keys() \
                if ('pos_token' in x and f'n{modality}_pos_token' not in x)])
            pos_tokens_vname = [f'{modality}_pos_token_{i}' for i in range(n_pos_codebooks)]

            br2load = tokens_vname + pos_tokens_vname

            if sort_by_var is not None:
                sort_by_var = sort_by_var.replace(modality, f'{modality}_inp')
                if sort_by_var not in tree.keys():
                    sort_by_var = sort_by_var.replace('_inp', '')
                br2load.append(sort_by_var)

            if ev_num_fp is not None:
                br2load.append('event_number')

            total_elem = 0
            total_events = 0

        counts = tree[f'n{modality}'].array(library='np')
        total_elem += np.sum(counts)
        total_events += counts.shape[0]

        f.close()

    log.info("%s: %d rows, %d content and %d position codebooks", modality, total_elem, n_codebooks, n_pos_codebooks)

    n_cols = n_codebooks + n_pos_codebooks


    # 2.1 combined Data (total_elem, total_cols)
    fp_data = np.memmap(f"{output_file_prefix}_data.npy", 
        dtype='int16', mode='w+', shape=(total_elem, n_cols))

    # 2.2 Offsets
    fp_offsets = np.memmap(f"{output_file_prefix}_offsets.npy", 
        dtype='int64', mode='w+', shape=(total_events + 1,))
    fp_offsets[0] = 0

    # 2.3 Event numbers
    if ev_num_fp is not None:
        fp_evnums = np.memmap(ev_num_fp, 
            dtype='int64', mode='w+', shape=(total_events,))


    # 3.0 parse and write
    current_offset = 0; current_elem = 0; current_ev_idx = 0

    # Use iterate to handle large files smoothly
    for fp_i, fp in tqdm(enumerate(filelist), desc="   Parsing Files", total=len(filelist)):
        
        # iterate in chunks of 200k events to bound memory
        iterator = uproot.iterate(
            f"{fp}:event_tree", 
            expressions=br2load, 
            library='ak', 
            step_size=200_000 
        )

        for chunk_dict in tqdm(iterator, desc=f"      File {fp_i+1}", leave=False):
            
            # 3.1b event numbers (loaded when ev_num_fp is given)
            if 'event_number' in chunk_dict.fields:
                chunk_evnums = ak.to_numpy(chunk_dict['event_number'])

            # 3.2 sort_by_var
            if sort_by_var is not None:
                sort_vals = chunk_dict[sort_by_var]
                sort_idxs = ak.argsort(sort_vals, axis=1, ascending=False)
                for k in tokens_vname + pos_tokens_vname:
                    chunk_dict[k] = chunk_dict[k][sort_idxs]

            # 3.3a repeat pos tokens if needed
            if repeat_pos_tokens:
                ref_var = tokens_vname[0]
                ref_counts = ak.num(chunk_dict[ref_var])

                for p_var in pos_tokens_vname:
                    p_counts = ak.num(chunk_dict[p_var])
                    
                    # Only broadcast if lengths differ
                    if ak.any(p_counts != ref_counts):
                        
                        # CASE 1: Global Info (Length 1 per event) -> Broadcast to N
                        if ak.all(p_counts == 1):
                            # 1. Extract the scalar value (drop the inner list dimension)
                            # [[239], [484]] -> [239, 484]
                            scalar_pos = chunk_dict[p_var][:, 0]
                            
                            # 2. Broadcast scalar against the reference structure
                            # This creates [[239, 239...], [484, 484...]]
                            _, broadcasted_pos = ak.broadcast_arrays(chunk_dict[ref_var], scalar_pos)
                            
                            chunk_dict[p_var] = broadcasted_pos

                        else:
                            raise ValueError(f"Cannot repeat pos tokens for variable {p_var}: incompatible lengths.")

            # 3.3 flatten and stacking
            flat_data = [ak.to_numpy(ak.flatten(chunk_dict[v])).astype(np.int16) \
                for v in tokens_vname + pos_tokens_vname]
            stacked_data = np.stack(flat_data, axis=1)

            # 3.4 offsets computation
            counts = ak.num(chunk_dict[tokens_vname[0]]).to_numpy()
            offsets = np.cumsum(counts)

            # 3.5 write to mmaps
            n_rows = stacked_data.shape[0]
            n_evs = len(counts)
            
            fp_data[current_elem : current_elem + n_rows, :] = stacked_data
            
            # Offsets are relative to the current chunk start!
            fp_offsets[current_offset + 1 : current_offset + 1 + n_evs] = offsets + current_elem
            
            # Write Event Numbers (Assuming you added it to br2load)
            if ev_num_fp is not None and 'event_number' in chunk_dict.fields:
                fp_evnums[current_ev_idx : current_ev_idx + n_evs] = ak.to_numpy(chunk_evnums)

            # 3.6 update cursors
            current_elem += n_rows
            current_offset += n_evs
            current_ev_idx += n_evs


    # save meta
    np.savez(f"{output_file_prefix}_meta.npz", 
        n_events=total_events, n_codebooks=n_codebooks, n_pos_codebooks=n_pos_codebooks)

    # close memmaps
    fp_data.flush()
    fp_offsets.flush()
