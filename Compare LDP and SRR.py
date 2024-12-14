import pandas as pd
import argparse
import dill
import copy
from tools.save_results import generate_CF_record,save_CF_to_csv,add_to_CF_file
from dpmechanisms.local_dp import Make_Private,Compare,make_DS_LDP
from utils.analysis.evaluation import reidentify,compare_solutions
from utils.data.distance import KLDivergence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='adult', help='hospital,adult,informs,synth_adult')
    parser.add_argument('--rseed', type=int, default=2, help='random seed: choose between 0 - 5')
    parser.add_argument('--model', type=str, default='RF', help='NN, RF, SVM, XgBoost')
    parser.add_argument('--n_count', type=int, default=3, help='3 5 10 20')
    parser.add_argument('--epsilon', type=float, default='10', help='.01, .1, 1, 5, 10')
    parser.add_argument('--CF_method', type=str, default='synth_dp_cf', help='NICE,zerocost_DP_CF,ldp_ds_cf,synth_dp_cf,LDP_CF,LDP_Noisy_max,inline_LDP,LDP_SRR,LDP_Noisy_max')
                                                                        
    parser.add_argument('--instance_num', type=int, default=1 , help='from 1 to 100')

    # get input      
    args = parser.parse_args()
    dataset_name = args.dataset
    RANDOM_SEED = args.rseed
    model = args.model
    epsilon = args.epsilon
    instance_num = args.instance_num
    n_count = args.n_count
    if epsilon >= 1:
        epsilon = int(epsilon)
    #input: NICE_model,CF_method,to_explain,DS,filepathname,epsilon=1,NEIGHBOR_COUNT=3,is_synthetic = False, Real_DS = None
    resultsdir =    './dpnice/cf_results/{}/'.format(dataset_name)       
    modeloutdir = './dpnice/pretrained/{}/'.format(dataset_name)
    DSoutdir = './dpnice/datasets_loaded/{}/'.format(dataset_name)
    synth_DSoutdir = './dpnice/datasets_loaded/synth_{}/'.format(dataset_name)
    CF_file_dir = './dpnice/cf_files/{}/'.format(dataset_name)

    resultfilename = '{}{}_{}_eps_{}.csv'.format(resultsdir, model, RANDOM_SEED,epsilon)   
    model_name = '{}{}_{}.pkl'.format(modeloutdir, model, RANDOM_SEED)
    DS_name = '{}{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED)
    synth_DS_name = '{}{}_{}_{}.pkl'.format(synth_DSoutdir, model, RANDOM_SEED,epsilon)

    with open(model_name, 'rb') as file:
        NICE_model = dill.load(file)
            #load the dataset
    with open(DS_name, 'rb') as file:
        DS = dill.load(file)
        #select original instance
    X_train = DS.X_train.copy()
    ldp_DS = copy.deepcopy(DS)
    ldp_DS.X_train = make_DS_LDP(X_train,DS.dataset['feature_ranges'],epsilon,ldp_mech=0,cat_type =0)


    # to_explain = DS.X_test[instance_num:instance_num+1 :] 
    KL_divergence_machine = KLDivergence(X_train)
    
    for instance_num in range(1,101):
        to_explain = DS.X_test[instance_num:instance_num+1 :] 
        for CF_method in ('NICE','zerocost_DP_CF','ldp_ds_cf','synth_dp_cf','LDP_CF','LDP_SRR','LDP_Noisy_max','inline_LDP'):
            CF_filename =  '{}{}_{}_eps_{}_{}_k_{}.csv'.format(CF_file_dir,model,RANDOM_SEED,epsilon,CF_method,n_count)
            Basic_instances_filename =  '{}{}_{}_eps_{}_{}_basic_k_{}.csv'.format(CF_file_dir,model,RANDOM_SEED,epsilon,CF_method,n_count)
            NICE_CF_filename = '{}{}_{}_eps_{}_NICE.csv'.format(CF_file_dir,model,RANDOM_SEED,epsilon)
            if CF_method == 'NICE':
                if(epsilon == 1):
                    prediction = NICE_model.data.predict_fn(to_explain).argmax() 
                    CF,basic_instance = NICE_model.explain_Second(to_explain)
                    add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'],ismember=True)
                    add_to_CF_file(NICE_CF_filename,CF[0], columns_list=DS.dataset['feature_names'])
                        ## if our private countefactual is yet a counterfactual
                    nice_df = pd.DataFrame(data=[to_explain[0], CF[0]], columns=DS.dataset['feature_names'])
                    
                    if (NICE_model.data.predict_fn(CF).argmax() == prediction):
                        no_CF = True 
                        print("for epsilon:",epsilon,"NICE:",CF,"is not Counterfactual")
                        record = generate_CF_record(CF_method, nice_df, CF_distance='NA', CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' ,cf__ri_count='NA', not_cf=no_CF, epsilon=0, k=0, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA')
                        save_CF_to_csv(record, csv_file_path=resultfilename)
                        
                    else:
                        no_CF = False
                        plaus_distances = NICE_model.Calculate_Plaus_Dists(CF)
                        CF_distance = NICE_model.distance_metric.measure(CF,to_explain)
                        print("for epsilon:",epsilon, "NICE:",nice_df,"Distance:",CF_distance)
                        # eq_to_basic = Compare(CF,ldp_cf)  #if original instance has been changed by ldp or not
                        # changes_rate, same_inst_rate = compare_solutions(to_explain,CF,ldp_cf)
                        cf__ri_count =reidentify(CF,X_train)
                        nice_KL_divergence = KL_divergence_machine.distance(basic_instance,CF)
                        record = generate_CF_record(CF_method, nice_df, CF_distance,plaus_distances[0],plaus_distances[1],plaus_distances[2], cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate='NA', same_inst_rate='NA',SameAsNice= 'NA',KL_divergence=nice_KL_divergence)
                        save_CF_to_csv(record, csv_file_path=resultfilename)
                        if cf__ri_count > 0 :
                            member = True
                        else: 
                            member = False
                        add_to_CF_file(CF_filename,CF[0], columns_list=DS.dataset['feature_names'],ismember=member)

            elif CF_method == 'zerocost_DP_CF':
                for n_count in(20,10,5,3):
                    CF_filename =  '{}{}_{}_eps_{}_{}_k_{}.csv'.format(CF_file_dir,model,RANDOM_SEED,epsilon,CF_method,n_count)
                    Basic_instances_filename =  '{}{}_{}_eps_{}_{}_basic_k_{}.csv'.format(CF_file_dir,model,RANDOM_SEED,epsilon,CF_method,n_count)
                    zerocost_DP_CF,zerocost_basic_CF,n_tries = NICE_model.private_explain_plausible_fail_stop(to_explain,change_cost = DS.dataset['zerocosts'],K=n_count,epsilon = epsilon)# private_explain(to_explain,change_cost = zerocosts,K=NEIGHBOR_COUNT)
                    print(f"The {instance_num}th iteration zerocost_DP_CF run")
                    df = pd.DataFrame(data=[to_explain[0], zerocost_DP_CF[0]], columns=DS.dataset['feature_names'])       
                    if(n_tries == 0):
                        no_CF = False
                        changes_rate, same_inst_rate = compare_solutions(to_explain,zerocost_basic_CF,zerocost_DP_CF)
                        eq_to_basic = Compare(zerocost_basic_CF,zerocost_DP_CF)  #if original instance has been changed by dp or not
                        CF_distance = NICE_model.distance_metric.measure(zerocost_DP_CF,to_explain)
                        plaus_distances = NICE_model.Calculate_Plaus_Dists(zerocost_DP_CF)
                        cf__ri_count  = reidentify(zerocost_DP_CF,X_train)
                        # print("DP_CF:",df,"Distance:",CF_distance)
                        zero_cost__KL_divergence = KL_divergence_machine.distance(zerocost_basic_CF,zerocost_DP_CF)
                        record = generate_CF_record(CF_method, df, CF_distance, plaus_distances[0],plaus_distances[1],plaus_distances[2], cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=n_count, changed_rate=changes_rate, SameAsNice=eq_to_basic,same_inst_rate=same_inst_rate,KL_divergence=zero_cost__KL_divergence) 
                        save_CF_to_csv(record, csv_file_path=resultfilename)
                        add_to_CF_file(Basic_instances_filename,zerocost_basic_CF[0], columns_list=DS.dataset['feature_names'],ismember=True)
                        if cf__ri_count > 0 :
                            member = True
                        else: 
                            member = False
                        add_to_CF_file(CF_filename,zerocost_DP_CF[0], columns_list=DS.dataset['feature_names'],ismember=member)

                    else:    #no zerocost_DP_CF
                        print("exp_prox could not find plausible instance")
                        no_CF = True
                        record = generate_CF_record(CF_method, df, CF_distance='NA',CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' , cf__ri_count='NA', not_cf=no_CF, epsilon=epsilon, k=n_count, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA',KL_divergence='NA')
                        save_CF_to_csv(record, csv_file_path=resultfilename)

            
            elif CF_method == 'LDP_CF':
                prediction = NICE_model.data.predict_fn(to_explain).argmax() 
                CF,basic_instance = NICE_model.explain_Second(to_explain)
                
                # add_to_CF_file(NICE_CF_filename,CF[0], columns_list=DS.dataset['feature_names'])
                ldp_cf = Make_Private(CF,to_explain,DS.dataset['feature_ranges'],DS.dataset['categorical_features'],DS.dataset['numerical_features'],epsilon,cat_type=1,ldp_Method=0) #value-wise(cat_type =1)  #naive_ldp(ldp_method = 0)
                    ## if our private countefactual is yet a counterfactual
                ldp_df = pd.DataFrame(data=[to_explain[0], ldp_cf[0]], columns=DS.dataset['feature_names'])
                
                if (NICE_model.data.predict_fn(ldp_cf).argmax() == prediction):
                    no_CF = True 
                    print("for epsilon:",epsilon,"ldp_CF:",ldp_cf,"is not Counterfactual")
                    record = generate_CF_record(CF_method, ldp_df, CF_distance='NA', CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' ,cf__ri_count='NA', not_cf=no_CF, epsilon=epsilon, k=0, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA')
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                    
                else:
                    no_CF = False
                    CF_distance = NICE_model.distance_metric.measure(ldp_cf,to_explain)
                    plaus_distances = NICE_model.Calculate_Plaus_Dists(ldp_cf)
                    # print("for epsilon:",epsilon, "ldp_CF:",ldp_df,"Distance:",CF_distance)
                    eq_to_basic = Compare(CF,ldp_cf)  #if original instance has been changed by ldp or not
                    changes_rate, same_inst_rate = compare_solutions(to_explain,CF,ldp_cf)
                    cf__ri_count = reidentify(ldp_cf,X_train)
                    ldp_KL_divergence = KL_divergence_machine.distance(basic_instance,ldp_cf)
                    record = generate_CF_record(CF_method, ldp_df, CF_distance, plaus_distances[0],plaus_distances[1],plaus_distances[2], cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate=changes_rate, same_inst_rate=same_inst_rate,SameAsNice= eq_to_basic,KL_divergence=ldp_KL_divergence)
                    add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'],ismember=True)
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                    if cf__ri_count > 0 :
                        member = True
                    else: 
                        member = False
                    add_to_CF_file(CF_filename,ldp_cf[0], columns_list=DS.dataset['feature_names'],ismember = member)
                    # add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'])


            elif CF_method == 'LDP_SRR':
                prediction = NICE_model.data.predict_fn(to_explain).argmax() 
                CF,basic_instance = NICE_model.explain_Second(to_explain)
                ldp_srr_cf = Make_Private(CF,to_explain,DS.dataset['feature_ranges'],DS.dataset['categorical_features'],DS.dataset['numerical_features'],epsilon,cat_type=1,ldp_Method=1) #SRR
                    ## if our private countefactual is yet a counterfactual
                ldp_srr_df = pd.DataFrame(data=[to_explain[0], ldp_srr_cf[0]], columns=DS.dataset['feature_names'])
                
                if (NICE_model.data.predict_fn(ldp_srr_cf).argmax() == prediction):
                    no_CF = True 
                    
                    # print("for epsilon:",epsilon,"LDP_SRR:",ldp_srr_cf,"is not Counterfactual")
                    record = generate_CF_record(CF_method, ldp_srr_df, CF_distance='NA',CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' , cf__ri_count='NA', not_cf=no_CF, epsilon=epsilon, k=0, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA')
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                else:
                    no_CF = False
                    CF_distance = NICE_model.distance_metric.measure(ldp_srr_cf,to_explain)
                    plaus_distances = NICE_model.Calculate_Plaus_Dists(ldp_srr_cf)
                    # print("for epsilon:",epsilon, "LDP_SRR:",ldp_srr_df,"Distance:",CF_distance)
                    eq_to_basic = Compare(CF,ldp_srr_cf)  #if original instance has been changed by ldp or not
                    changes_rate, same_inst_rate = compare_solutions(to_explain,CF,ldp_srr_cf)
                    cf__ri_count =reidentify(ldp_srr_cf,X_train)
                    SRR_KL_divergence = KL_divergence_machine.distance(basic_instance,ldp_srr_cf)
                    record = generate_CF_record(CF_method, ldp_srr_df, CF_distance, plaus_distances[0],plaus_distances[1],plaus_distances[2], cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate=changes_rate, same_inst_rate=same_inst_rate,SameAsNice= eq_to_basic,KL_divergence=SRR_KL_divergence)
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                    if cf__ri_count > 0 :
                        member = True
                    else: 
                        member = False
                    add_to_CF_file(CF_filename,ldp_srr_cf[0], columns_list=DS.dataset['feature_names'],ismember=member)
                    add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'],ismember= True)

            elif CF_method == 'LDP_Noisy_max':
                prediction = NICE_model.data.predict_fn(to_explain).argmax() 
                CF,basic_instance = NICE_model.explain_Second(to_explain)
                noisy_max_cf = Make_Private(CF,to_explain,DS.dataset['feature_ranges'],DS.dataset['categorical_features'],DS.dataset['numerical_features'],epsilon,dataset=NICE_model.data.candidates_view,cat_type=1,ldp_Method=2) #value-wise
                    ## if our private countefactual is yet a counterfactual
                noisy_max_df = pd.DataFrame(data=[to_explain[0], noisy_max_cf[0]], columns=DS.dataset['feature_names'])
                
                if (NICE_model.data.predict_fn(noisy_max_cf).argmax() == prediction):
                    no_CF = True 
                    
                    print("for epsilon:",epsilon,"noisy_max:",noisy_max_cf,"is not Counterfactual")
                    record = generate_CF_record(CF_method, noisy_max_df, CF_distance='NA',CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' , cf__ri_count='NA', not_cf=no_CF, epsilon=epsilon, k=0, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA')
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                else:
                    no_CF = False
                    CF_distance = NICE_model.distance_metric.measure(noisy_max_cf,to_explain)
                    plaus_distances = NICE_model.Calculate_Plaus_Dists(noisy_max_cf)
                    # print("for epsilon:",epsilon, "noisy_max:",noisy_max_df,"Distance:",CF_distance)
                    eq_to_basic = Compare(CF,noisy_max_cf)  #if original instance has been changed by ldp or not
                    changes_rate, same_inst_rate = compare_solutions(to_explain,CF,noisy_max_cf)
                    cf__ri_count =reidentify(noisy_max_cf,X_train)
                    Noisy_max_KL_divergence = KL_divergence_machine.distance(basic_instance,noisy_max_cf)
                    record = generate_CF_record(CF_method, noisy_max_df, CF_distance, plaus_distances[0],plaus_distances[1],plaus_distances[2], cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate=changes_rate, same_inst_rate=same_inst_rate,SameAsNice= eq_to_basic,KL_divergence=Noisy_max_KL_divergence)
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                    if cf__ri_count > 0 :
                        member = True
                    else: 
                        member = False
                    add_to_CF_file(CF_filename,noisy_max_cf[0], columns_list=DS.dataset['feature_names'],ismember=member)
                    add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'],ismember=True)
            
            elif CF_method == 'inline_LDP':
                prediction = NICE_model.data.predict_fn(to_explain).argmax() 
                for n_count in(20,10,5,3):
                    CF_filename =  '{}{}_{}_eps_{}_{}_k_{}.csv'.format(CF_file_dir,model,RANDOM_SEED,epsilon,CF_method,n_count)
                    Basic_instances_filename =  '{}{}_{}_eps_{}_{}_basic_k_{}.csv'.format(CF_file_dir,model,RANDOM_SEED,epsilon,CF_method,n_count)
                    # inline_LDP_CF,basic_instance,n_try = NICE_model.private_explain_LDP(to_explain,K = n_count,feature_ranges =DS.dataset['feature_ranges'],epsilon=epsilon,ldp_mech = 0) #for now, naive implementation
                    inline_LDP_CF,basic_instance,n_try = NICE_model.private_explain_LDP_early_stop(to_explain,K = n_count,feature_ranges =DS.dataset['feature_ranges'],epsilon=epsilon,ldp_mech = 0)
                    # noisy_max_cf = Make_Private(CF,to_explain,DS.dataset['feature_ranges'],DS.dataset['categorical_features'],DS.dataset['numerical_features'],epsilon,dataset=NICE_model.data.candidates_view,cat_type=1,ldp_Method=2) #value-wise
                        ## if our private countefactual is yet a counterfactual
                    inline_LDP_df = pd.DataFrame(data=[to_explain[0], inline_LDP_CF[0]], columns=DS.dataset['feature_names'])
                    
                    if (NICE_model.data.predict_fn(inline_LDP_CF).argmax() == prediction or  (NICE_model.data.predict_fn(inline_LDP_CF).argmax() != prediction and n_try>0)):
                        no_CF = True 
                        
                        print("for epsilon:",epsilon,"inline_LDP_CF:",inline_LDP_CF,"is not Counterfactual")
                        record = generate_CF_record(CF_method, inline_LDP_df, CF_distance='NA', CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' ,cf__ri_count='NA', not_cf=no_CF, epsilon=epsilon, k=n_count, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA')
                        save_CF_to_csv(record, csv_file_path=resultfilename)
                    else:
                        no_CF = False
                        CF_distance = NICE_model.distance_metric.measure(inline_LDP_CF,to_explain)
                        plaus_distances = NICE_model.Calculate_Plaus_Dists(inline_LDP_CF)
                        # print("for epsilon:",epsilon, "inline_LDP_CF:",inline_LDP_df,"Distance:",CF_distance)
                        eq_to_basic = Compare(basic_instance,inline_LDP_CF)  #if original instance has been changed by ldp or not
                        changes_rate, same_inst_rate = compare_solutions(to_explain,basic_instance,inline_LDP_CF)
                        cf__ri_count =reidentify(inline_LDP_CF,X_train)
                        inline_LDP_KL_divergence = KL_divergence_machine.distance(basic_instance,inline_LDP_CF)
                        record = generate_CF_record(CF_method, inline_LDP_df, CF_distance,CF_min_dist=plaus_distances[0],CF_min_k_dist=plaus_distances[1],CF_rand_k_dist=plaus_distances[2], cf__ri_count= cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=n_count, changed_rate=changes_rate, same_inst_rate=same_inst_rate,SameAsNice= eq_to_basic,KL_divergence=inline_LDP_KL_divergence)
                        save_CF_to_csv(record, csv_file_path=resultfilename)
                        if cf__ri_count > 0 :
                            member = True
                        else: 
                            member = False
                        add_to_CF_file(CF_filename,inline_LDP_CF[0], columns_list=DS.dataset['feature_names'],ismember = member)
                        add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'],ismember=True)
            
            elif CF_method == 'synth_dp_cf':
                with open(synth_DS_name, 'rb') as file:
                        Synth_DS = dill.load(file)

                # X_train = Synth_DS.X_train.copy()

                prediction = NICE_model.data.predict_fn(to_explain).argmax() 
                # inline_LDP_CF,basic_instance,n_try = NICE_model.private_explain_LDP(to_explain,K = n_count,feature_ranges =DS.dataset['feature_ranges'],epsilon=epsilon,ldp_mech = 0) #for now, naive implementation
                synth_CF,basic_instance,n_try = NICE_model.synth_explain(to_explain,DS = Synth_DS)
                # noisy_max_cf = Make_Private(CF,to_explain,DS.dataset['feature_ranges'],DS.dataset['categorical_features'],DS.dataset['numerical_features'],epsilon,dataset=NICE_model.data.candidates_view,cat_type=1,ldp_Method=2) #value-wise
                    ## if our private countefactual is yet a counterfactual
                synth_df = pd.DataFrame(data=[to_explain[0], synth_CF[0]], columns=DS.dataset['feature_names'])
                
                if (NICE_model.data.predict_fn(synth_CF).argmax() == prediction or  (NICE_model.data.predict_fn(synth_CF).argmax() != prediction and n_try>0)):
                    no_CF = True 
                    
                    print("for epsilon:",epsilon,"synth_CF:",synth_CF,"is not Counterfactual")
                    record = generate_CF_record(CF_method, synth_CF, CF_distance='NA',CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' , cf__ri_count='NA', not_cf=no_CF, epsilon=epsilon, k=0, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA')
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                else:
                    no_CF = False
                    plaus_distances = NICE_model.Calculate_Plaus_Dists(synth_CF,should_fit = True, X = to_explain)
                    CF_distance = NICE_model.distance_metric.measure(synth_CF,to_explain)
                    print("for epsilon:",epsilon, "Synth_CF:",synth_df,"Distance:",CF_distance)
                    eq_to_basic = Compare(basic_instance,synth_CF)  #if original instance has been changed by ldp or not
                    changes_rate, same_inst_rate = compare_solutions(to_explain,basic_instance,synth_CF)
                    cf__ri_count =reidentify(synth_CF,X_train)
                    synth_CF_KL_divergence = KL_divergence_machine.distance(basic_instance,synth_CF)
                    record = generate_CF_record(CF_method, synth_df, CF_distance, CF_min_dist=plaus_distances[0],CF_min_k_dist=plaus_distances[1],CF_rand_k_dist=plaus_distances[2], cf__ri_count=cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate=changes_rate, same_inst_rate=same_inst_rate,SameAsNice= eq_to_basic,KL_divergence=synth_CF_KL_divergence)
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                    if cf__ri_count > 0 :
                        member = True
                    else: 
                        member = False
                    add_to_CF_file(CF_filename,synth_CF[0], columns_list=DS.dataset['feature_names'],ismember=member)
                    synth_basic_count =reidentify(basic_instance,X_train)
                    add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'],ismember=(synth_basic_count>0))

            elif CF_method == 'ldp_ds_cf':
                # ldp_DS = make_ldp(X_train,epsilon)

                prediction = NICE_model.data.predict_fn(to_explain).argmax() 
                # inline_LDP_CF,basic_instance,n_try = NICE_model.private_explain_LDP(to_explain,K = n_count,feature_ranges =DS.dataset['feature_ranges'],epsilon=epsilon,ldp_mech = 0) #for now, naive implementation
                ldp_ds_cf,basic_instance,n_try = NICE_model.synth_explain(to_explain,DS = ldp_DS)
                # noisy_max_cf = Make_Private(CF,to_explain,DS.dataset['feature_ranges'],DS.dataset['categorical_features'],DS.dataset['numerical_features'],epsilon,dataset=NICE_model.data.candidates_view,cat_type=1,ldp_Method=2) #value-wise
                    ## if our private countefactual is yet a counterfactual
                ldp_ds_df = pd.DataFrame(data=[to_explain[0], ldp_ds_cf[0]], columns=DS.dataset['feature_names'])
                
                if (NICE_model.data.predict_fn(ldp_ds_cf).argmax() == prediction or  (NICE_model.data.predict_fn(ldp_ds_cf).argmax() != prediction and n_try>0)):
                    no_CF = True 
                    
                    print("for epsilon:",epsilon,"LDP_DS_CF:",ldp_ds_cf,"is not Counterfactual")
                    record = generate_CF_record(CF_method, ldp_ds_cf, CF_distance='NA', CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' ,cf__ri_count='NA', not_cf=no_CF, epsilon=epsilon, k=0, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA')
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                else:
                    no_CF = False
                    CF_distance = NICE_model.distance_metric.measure(ldp_ds_cf,to_explain)
                    plaus_distances = NICE_model.Calculate_Plaus_Dists(ldp_ds_cf,should_fit = True, X = to_explain)
                    # print("for epsilon:",epsilon, "LDP_DS_CF:",ldp_ds_df,"Distance:",CF_distance)
                    eq_to_basic = Compare(basic_instance,ldp_ds_cf)  #if original instance has been changed by ldp or not
                    changes_rate, same_inst_rate = compare_solutions(to_explain,basic_instance,ldp_ds_cf)
                    cf__ri_count =reidentify(ldp_ds_cf,X_train)
                    ldp_ds_cf_KL_divergence = KL_divergence_machine.distance(basic_instance,ldp_ds_cf)
                    record = generate_CF_record(CF_method, ldp_ds_df, CF_distance, CF_min_dist= plaus_distances[0],CF_min_k_dist=plaus_distances[1],CF_rand_k_dist=plaus_distances[2], cf__ri_count= cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate=changes_rate, same_inst_rate=same_inst_rate,SameAsNice= eq_to_basic,KL_divergence=ldp_ds_cf_KL_divergence)
                    save_CF_to_csv(record, csv_file_path=resultfilename)
                    
                    add_to_CF_file(CF_filename,ldp_ds_cf[0], columns_list=DS.dataset['feature_names'],ismember=(cf__ri_count>0))
                    add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'],ismember= (reidentify(basic_instance,X_train)>0))