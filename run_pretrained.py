import csv

from tqdm import tqdm

from src.model.pretrained import CNNLoader
from src.model.trainer import Trainer
from src.datasets.data import Data
from src.adversarial.cw_test import CWTestEnvironment
from src.adversarial.auc_test import AUCTest
from options import args

def run_setup(args, model, input_size, n_classes, model_dir_name):
    data = Data(dataset_name=args.dataset,
                device=args.device,
                batch_size=args.batchsize,
                transform=args.transform,
                model=model,
                input_size=input_size,
                adversarial_opt=args.adversarial_opt,
                adversarial_training_opt=args.adversarial_training_opt,
                n_datapoints=args.n_datapoints,
                jpeg_compression=args.jpeg_compression,
                jpeg_compression_rate=args.jpeg_compression_rate)
    
    trainer = Trainer(opt=args,
                    model=model,
                    data=data.dataset,
                    model_name=model_dir_name,
                    num_classes=n_classes,
                    optim_args=args.optim,
                    epochs=args.epochs,
                    model_type=args.model_out,
                    log_result=args.log_result,
                    adversarial_training_opt=args.adversarial_training_opt)
    
    trainer.test_model()
    path_to_run_results = trainer.training.utils.logger.run_name
    return path_to_run_results




if __name__ == '__main__':
    
    if args.dataset == 'nips17':
        n_classes = 1000
        model_dir_name = 'ImgNetCNN'
    elif args.dataset == 'cifar10':
        n_classes = 10
        model_dir_name = 'CIFARCNN'
    elif args.dataset == 'cifar100':
        n_classes = 100
        model_dir_name = 'CIFAR100CNN'
    else:
        n_classes = 2
        model_dir_name = 'FlickrCNN'

    loader = CNNLoader(args.pretrained, args.adversarial_pretrained_opt)
    cnn, input_size = loader.transfer(args.model_name,
                                      n_classes, 
                                      feature_extract=args.as_ft_extractor, 
                                      device=args.device)
    cnn.model_name = args.model_name
    model_dir_name += '_' + cnn.model_name
    
    if args.run_cw_test:
        
        cw_test = CWTestEnvironment(cw_type=args.adversarial_opt.spatial_adv_type,
                                    model=cnn, 
                                    optim_steps=args.steps, 
                                    n_datapoints=args.n_datapoints,
                                    target_mode=args.target_mode,
                                    dataset_type=args.dataset,
                                    test_robustness=args.test_robustness,
                                    iq_loss=args.iq_loss)
        if cw_test.test_robustness:
            
            args.adversarial_opt.attack_compression = True
            
            for compression_val in cw_test.compression_range:
                
                print(f'testing for compression:{compression_val}\n')
            
                args.adversarial_opt.compression_rate = compression_val
                path_compr_dir = cw_test.get_compr_dir(compression_val=compression_val)
                
                            
                for config in tqdm(cw_test.config_list):
                    cw_test.reset()
                    
                    c = config[0]
                    attack_lr = config[1]
                    c = c.item()
                    
                    print('\n##############################################################\n')
                    print('Initialize new configuration:\n\n')
                    print(f'\t c: {c}\n')
                    print(f'\t attack_lr: {attack_lr}\n')
                    
                    run_csv = f'{path_compr_dir}/loss_reports/report_c-{c}_lr-{attack_lr}.csv'
                    
                    with open(run_csv, 'a') as report_file:
                        report_obj = csv.writer(report_file)
                        report_obj.writerow(['file', 'iq_loss', 'f_loss', 'cost'])
                    
                    # 1st value of config is always c, 2nd is always attack_lr
                    args.adversarial_opt.spatial_attack_params.c = c
                    args.adversarial_opt.spatial_attack_params.attack_lr = attack_lr
                    args.adversarial_opt.spatial_attack_params.steps = cw_test.optim_steps
                    args.adversarial_opt.spatial_attack_params.n_starts = cw_test.n_starts
                    args.adversarial_opt.spatial_attack_params.write_protocol = True
                    args.adversarial_opt.spatial_attack_params.protocol_file = run_csv

                    path_to_run_results = run_setup(args, model=cnn, input_size=input_size, n_classes=n_classes, model_dir_name=model_dir_name)
                    asp = cw_test.write_to_protocol_dir(c=c, attack_lr=attack_lr, run_csv=run_csv, compression=True, logger_run_file=path_to_run_results)
                    cw_test.asp_list.append(asp)
                
                c_list, asp_list =  cw_test.get_avg_asp_per_c()
                cw_test.auc_metric(c_list, asp_list)    
                cw_test.plot_cw_test_results(compression_val=compression_val)
        else:
            
            for config in tqdm(cw_test.config_list):
                
                cw_test.reset()
                
                c = config[0]
                attack_lr = config[1]
                c = c.item()
                
                print('\n##############################################################\n')
                print('Initialize new configuration:\n\n')
                print(f'\t c: {c}\n')
                print(f'\t attack_lr: {attack_lr}\n')
                
                run_csv = f'{cw_test.report_dir}/loss_reports/report_c-{c}_lr-{attack_lr}.csv'
                
                with open(run_csv, 'a') as report_file:
                    report_obj = csv.writer(report_file)
                    report_obj.writerow(['file', 'iq_loss', 'f_loss', 'cost'])
                
                # 1st value of config is always c, 2nd is always attack_lr
                args.adversarial_opt.spatial_attack_params.c = c
                args.adversarial_opt.spatial_attack_params.attack_lr = attack_lr
                args.adversarial_opt.spatial_attack_params.steps = cw_test.optim_steps
                args.adversarial_opt.spatial_attack_params.n_starts = cw_test.n_starts
                args.adversarial_opt.spatial_attack_params.write_protocol = True
                args.adversarial_opt.spatial_attack_params.protocol_file = run_csv

                path_to_run_results = run_setup(args, model=cnn, input_size=input_size, n_classes=n_classes, model_dir_name=model_dir_name)
                asp = cw_test.write_to_protocol_dir(c=c, attack_lr=attack_lr, run_csv=run_csv, compression=False, logger_run_file=path_to_run_results)
                cw_test.asp_list.append(asp)

            c_list, asp_list =  cw_test.get_avg_asp_per_c()
            cw_test.auc_metric(c_list, asp_list)
            cw_test.plot_cw_test_results()

    elif args.run_auc_test:

        auc_test = AUCTest(attack_type=args.adversarial_opt.spatial_adv_type,
                                    model=cnn,
                                    dataset_type=args.dataset,
                                    test_robustness=args.test_robustness,
                                    spatial_attack_params=args.adversarial_opt.spatial_attack_params,
                                    c=args.c,
                                    lr=args.lr)
        if auc_test.test_robustness:
            
            args.adversarial_opt.attack_compression = True
            
            for compression_val in auc_test.compression_range:
                auc_test.reset()
                
                print(f'testing for compression:{compression_val}\n')
            
                args.adversarial_opt.compression_rate = compression_val
                path_compr_dir = auc_test.get_compr_dir(compression_val=compression_val)
                auc_test.auc_metric.update_target_path(f'{path_compr_dir}/auc_result.txt')
                
                if auc_test.attack_family == 'adv_optim':
                        print('\n##############################################################\n')
                        print('Initialize CW AUC test\n\n')

                        run_csv = f'{path_compr_dir}/reports/report.csv'
                        # cw tests need only one run per epsilon as they are invariant to it

                        with open(run_csv, 'a') as report_file: 
                            report_obj = csv.writer(report_file)
                            report_obj.writerow(['file', 'eps', 'asp'])

                        path_to_run_results = run_setup(args, model=cnn, input_size=input_size, n_classes=n_classes, model_dir_name=model_dir_name)
                        asp = auc_test.write_to_protocol_dir(eps=None, run_csv=run_csv, compression=True, logger_run_file=path_to_run_results)
                        auc_test.plot_roc(is_adv_optim_run=True, compression_val=compression_val)
                else:
                    for eps_value in tqdm(auc_test.eps_list):
                        
                        
                        print('\n##############################################################\n')
                        print('Initialize new configuration:\n\n')
                        print(f'\t eps: {eps_value}\n')
                        
                        
                        run_csv = f'{path_compr_dir}/reports/report_eps-{eps_value}.csv'
                        
                        with open(run_csv, 'a') as report_file:
                            report_obj = csv.writer(report_file)
                            report_obj.writerow(['file', 'eps', 'asp'])
                        
                        
                        #args.adversarial_opt.spatial_attack_params.write_protocol = True
                        #args.adversarial_opt.spatial_attack_params.protocol_file = run_csv
                        args.adversarial_opt.spatial_attack_params.eps = eps_value

                        path_to_run_results = run_setup(args, model=cnn, input_size=input_size, n_classes=n_classes, model_dir_name=model_dir_name)
                        asp = auc_test.write_to_protocol_dir(eps=eps_value, run_csv=run_csv, compression=True, logger_run_file=path_to_run_results)
                        auc_test.asp_list.append(asp)
                        
                    auc_test.auc_metric(auc_test.eps_list, auc_test.asp_list)    
                    auc_test.plot_roc(compression_val=compression_val)
        else:
            if auc_test.attack_family == 'adv_optim':
                print('\n##############################################################\n')
                print('Initialize CW AUC test\n\n')

                # cw tests need only one run per epsilon as they are invariant to it
                run_csv = f'{auc_test.report_dir}/reports/report.csv'

                with open(run_csv, 'a') as report_file: 
                    report_obj = csv.writer(report_file)
                    report_obj.writerow(['file', 'eps', 'asp'])

                path_to_run_results = run_setup(args, model=cnn, input_size=input_size, n_classes=n_classes, model_dir_name=model_dir_name)
                asp = auc_test.write_to_protocol_dir(eps=None, run_csv=run_csv, compression=True, logger_run_file=path_to_run_results)
                auc_test.plot_roc(is_adv_optim_run=True)
            else:
                for eps_value in tqdm(auc_test.eps_list):
                        
                    print('\n##############################################################\n')
                    print('Initialize new configuration:\n\n')
                    print(f'\t eps: {eps_value}\n')
                    
                    
                    run_csv = f'{auc_test.report_dir}/reports/report_eps-{eps_value}.csv'
                    
                    with open(run_csv, 'a') as report_file:
                        report_obj = csv.writer(report_file)
                        report_obj.writerow(['file', 'eps', 'asp'])
                    
                    
                    #args.adversarial_opt.spatial_attack_params.write_protocol = True
                    #args.adversarial_opt.spatial_attack_params.protocol_file = run_csv
                    args.adversarial_opt.spatial_attack_params.eps = eps_value

                    path_to_run_results = run_setup(args, model=cnn, input_size=input_size, n_classes=n_classes, model_dir_name=model_dir_name)
                    asp = auc_test.write_to_protocol_dir(eps=eps_value, run_csv=run_csv, compression=False, logger_run_file=path_to_run_results)
                    auc_test.asp_list.append(asp)
                    
                auc_test.auc_metric(auc_test.eps_list, auc_test.asp_list)    
                auc_test.plot_roc()
            
    else:

        data = Data(dataset_name=args.dataset,
                    device=args.device,
                    batch_size=args.batchsize,
                    transform=args.transform,
                    model=cnn,
                    input_size=input_size,
                    adversarial_opt=args.adversarial_opt,
                    adversarial_training_opt=args.adversarial_training_opt,
                    n_datapoints=args.n_datapoints,
                    jpeg_compression=args.jpeg_compression,
                    jpeg_compression_rate=args.jpeg_compression_rate)

        args.model_dir_name += '_' + cnn.model_name
        trainer = Trainer(opt=args,
                        model=cnn,
                        data=data.dataset,
                        model_name=model_dir_name,
                        num_classes=n_classes,
                        optim_args=args.optim,
                        epochs=args.epochs,
                        model_type=args.model_out,
                        log_result=args.log_result,
                        adversarial_training_opt=args.adversarial_training_opt)

        if not args.pretrained and data.dataset.dataset_type not in ['nips17', 'cifar10']:
            #print(f'\nrunning training for: \n{trainer.training.model}\n')
            best_acc = trainer.train_model(args.save_opt)
         
        trainer.test_model()
        #avg_salient_mask = data.dataset.post_transforms.attack.attack.get_avg_salient_mask()



print('done')