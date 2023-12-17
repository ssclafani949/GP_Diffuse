import numpy as np
import tables
import pandas as pd
import glob
import argparse



def load_data(systematics):
    sim_nc = {}
    sim_cc = {}
    file_dir = '/data/user/ssclafani/PointSource/DNNCascade/i3_processing/version-1.0/DNNCascadeL4__i3_to_hdf5/Lepton-injector/'
    if systematics:
        dataset_id = [ '22430', '22431', '22432', '22437', '22448', '22457', '22458', '22459']
    else:
        dataset_id = [ '22492', '22493', '22494', '22495', '22496', '22497', '22498', '22499', '22500' ]


    for dataset in dataset_id:
        print('Loading dataset {}'.format(dataset))
        sim_cc[dataset] = tables.open_file(file_dir + dataset + f'/DNNCascadeL4__i3_to_hdf5_Lepton-injector_CC_{dataset}_00000000.hdf5')
        sim_nc[dataset] = tables.open_file(file_dir + dataset + f'/DNNCascadeL4__i3_to_hdf5_Lepton-injector_NC_{dataset}_00000000.hdf5')
    weights_cc = {}
    weights_nc = {}
    for dataset in dataset_id:
        weights_cc[dataset] = pd.read_hdf(f'/data/user/ssclafani/GP_Diffuse/weights/level4/weight_df_0{dataset}_cascade_CC.hdf')
        weights_nc[dataset] = pd.read_hdf(f'/data/user/ssclafani/GP_Diffuse/weights/level4/weight_df_0{dataset}_cascade_NC.hdf')
    return weights_cc, weights_nc, sim_cc, sim_nc

def make_cuts(weights, dataset, merge):

    cut_weights = {} 
    dataset_id = weights.keys()
    for id in dataset_id:
         
        energy_cut = (dataset[id].root.EventGeneratorSelectedRecoNN_I3Particle.cols.energy[:] > 500)
        # MuonBDT cut
        bdt1 = (dataset[id].root.BDT_bdt_max_depth_4_n_est_2000lr_0_02_seed_3_train_size_50.cols.pred_001[:] < 5e-3)
        # CascadeBDT cut
        bdt2 = (dataset[id].root['BDT_bdt_max_depth_4_n_est_1000lr_0.01_seed_3_train_size_50'].cols.pred_001[:] > 0.1)

        mask = bdt1 & bdt2 & energy_cut    
        print('Cutting for Dataset {} : {} / {} events'.format(id, sum(mask), len(energy_cut)))

        cut_weights[id] =  weights[id][mask]
        #print(cut_weights)
    if merge:
        result = pd.concat(cut_weights.values(), ignore_index=True)
    else:
        result = cut_weights
    return result


def add_status_exists(df, merge):
    if merge:
        df['reco_energy_exists'] = np.ones_like(df['nu_energy'])
        df['reco_dir_exists'] = np.ones_like(df['nu_energy'])
        df['reco_energy_fit_status'] = np.zeros_like(df['nu_energy'])
        df['reco_dir_fit_status'] = np.zeros_like(df['nu_energy'])
        
        #time = np.random.uniform(size = len(df['nu_energy'].values), low =55562, high = 55927)
        #from icecube import astro
        ra, dec = df['reco_azimuth'], df['reco_zenith'] - np.pi/2 #astro.dir_to_equa(df['reco_zenith'], df['reco_azimuth'], time)
        df['reco_ra'] = ra
        df['reco_dec'] = dec
        df['nu_dec'] = df['nu_zenith'] - np.pi/2
        df['nu_ra'] = df['nu_azimuth']
        df['powerlaw'] = df['weight1.0_astro']
        df['fluxless_weight'] = 1. * df['oneweight'] * (df['nu_energy']/1e5) ** 1
        print(df.keys())

        df['mceq_conv_H4a_SIBYLL23c'] = df['weight1.0_atmo']
    else:
        orig_keys = df.keys()
        for key in orig_keys:
            df[key]['reco_energy_exists'] = np.ones_like(df[key]['nu_energy'])
            df[key]['reco_dir_exists'] = np.ones_like(df[key]['nu_energy'])
            df[key]['reco_energy_fit_status'] = np.zeros_like(df[key]['nu_energy'])
            df[key]['reco_dir_fit_status'] = np.zeros_like(df[key]['nu_energy'])
            
            time = np.random.uniform(size = len(df[key]['nu_energy'].values), low =55562, high = 55927)
            df[key]['mjd_time_start'] = time
            #from icecube import astro
            ra, dec = df[key]['reco_azimuth'], df[key]['reco_zenith'] - np.pi/2 #astro.dir_to_equa(df['reco_zenith'], df['reco_azimuth'], time)
            df[key]['dataset'] = df[key]['dataset'].astype('int32')
            df[key]['Run'] = df[key]['run'].astype('int32')
            df[key]['Event'] = df[key]['event_id'].astype('int32')
            df[key]['SubEvent'] = df[key]['sub_event_id'].astype('int32')
        
            df[key]['reco_ra'] = ra
            df[key]['reco_dec'] = dec
            df[key]['nu_dec'] = df[key]['nu_zenith'] - np.pi/2
            df[key]['nu_ra'] = df[key]['nu_azimuth']
            df[key]['powerlaw'] = df[key]['weight1.0_astro']
            df[key]['fluxless_weight'] = 1. * df[key]['oneweight'] * (df[key]['nu_energy']/1e5) ** 1
            df[key]['mceq_conv_H4a_SIBYLL23c'] = df[key]['weight1.0_atmo']
        
    return df
if __name__ == '__main__':
    import argparse
    systematics = False 
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge', default=False)
    #parser.add_argument('--data_dir', help='directry of level 4 files')
    args = parser.parse_args()
     
    weights_cc, weights_nc, sim_cc, sim_nc = load_data(systematics)
    cut_weight_cc = make_cuts(weights_cc, sim_cc, args.merge)
    cut_weight_nc = make_cuts(weights_cc, sim_cc, args.merge)
    if args.merge :
        print('merging')
        df = pd.concat([cut_weight_cc, cut_weight_nc])
        df = add_status_exists(df, args.merge)
        df.to_hdf('/data/user/ssclafani/GP_Diffuse/NNMFit/datasets/baseline/mc_merged.hdf5', 'df')
    else:
        print('CC')
        df1 = add_status_exists(cut_weight_cc, args.merge)
        #import IPython
        #IPython.embed() 
        df2 =add_status_exists(cut_weight_nc, args.merge)
        if systematics:
            base_dir = 'systematics'
        else:
            base_dir = 'baseline'
        for key in df1.keys():
            df1[key].to_hdf('/data/user/ssclafani/GP_Diffuse/NNMFit/datasets/{}/{}_CC_DNNCascade_FTP.hdf5'.format(base_dir, key), 'df')
            df2[key].to_hdf('/data/user/ssclafani/GP_Diffuse/NNMFit/datasets/{}/{}_NC_DNNCascade_FTP.hdf5'.format(base_dir, key), 'df')
            import h5py
            print(df1[key].keys())
            filename = '/data/user/ssclafani/GP_Diffuse/NNMFit/datasets/{}/table/{}_CC_DNNCascade_FTP_table.hdf5'.format(base_dir, key)
            reco_columns = ['Run', 'Event', 'SubEvent', 'dataset', 'reco_ra', 'reco_dec', 
                            'reco_energy', 'reco_azimuth', 'reco_zenith', 'reco_energy_exists', 'reco_dir_exists', 'reco_energy_fit_status',
                            'reco_dir_fit_status' , 'mjd_time_start']
            mc_columns = ['Run', 'Event', 'SubEvent', 'oneweight', 'pdg', 'nu_azimuth', 'nu_zenith', 'nu_energy','fluxless_weight', 'mceq_conv_H4a_SIBYLL23c']
            f = h5py.File(filename, 'w')
            reco_dict = df1[key][reco_columns].to_records()
            mc_dict = df1[key][mc_columns].to_records()
            #import IPython
            #IPython.embed()
            f.create_dataset('reco', data=reco_dict)
            f.create_dataset('mc', data=mc_dict)
            f.close() 
            #reco_dict = df1[key][reco_columns]
            #reco_dict.to_hdf(filename, key='reco', format='fixed')
            #mc_dict = df1[key][mc_columns]
            #mc_dict.to_hdf(filename, key='mc', format='fixed')
            
            filename = '/data/user/ssclafani/GP_Diffuse/NNMFit/datasets/{}/table/{}_NC_DNNCascade_FTP_table.hdf5'.format(base_dir, key)
            f = h5py.File(filename, 'w')
            reco_dict = df2[key][reco_columns].to_records()
            mc_dict = df2[key][mc_columns].to_records()
            f.create_dataset('reco', data=reco_dict)
            f.create_dataset('mc', data=mc_dict)
            f.close() 
            #reco_dict = df2[key][reco_columns]
            #reco_dict.to_hdf(filename, key='reco', format='fixed')
            #mc_dict = df2[key][mc_columns]
            #mc_dict.to_hdf(filename, key='mc', format='fixed')
            tables.file._open_files.close_all()

    #df.to_parquet('/data/user/ssclafani/GP_Diffuse/weights/level5/mc_merged_parquet.parq', object_encoding='int', engine='fastparquet')

