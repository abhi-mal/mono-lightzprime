#!/usr/bin/env python
import os
from subprocess import Popen,PIPE,STDOUT
from datetime import date

today = date.today()
date_string = today.strftime("%y_%m_%d")
cwd = os.getcwd()

script = "Analyzer_DoubleMuCR.py"
#script ="Analyzer.py"
input_path_base = "/hdfs/store/user/mallampalli/analysis/Mono-zprime/nanoaod_tests/"
#output_path_base = "/nfs_scratch/mallampalli/Analysis/Mono-zprime/nanoAOD_test/analysis_trial/CMSSW_10_6_30/src/analysis/2018/SignalRegion/"
#output_path_base = "/nfs_scratch/mallampalli/Analysis/Mono-zprime/nanoAOD_test/analysis_trial/CMSSW_10_6_30/src/analysis/2018/DoubleMuCR/"
#output_path_base = "/hdfs/store/user/mallampalli/analysis/Mono-zprime/nanoaod_postfiles/using_nanoaodtools/2018/SignalRegion/"
output_path_base = "/hdfs/store/user/mallampalli/analysis/Mono-zprime/nanoaod_postfiles/using_nanoaodtools/2018/DoubleMuCR/"
cmssw_path = "/nfs_scratch/mallampalli/Analysis/Mono-zprime/nanoAOD_test/analysis_trial/CMSSW_10_6_30/"
year = output_path_base.split("/")[-3]

def make_needed_output_dirs(output_subdir):
        os.chdir(output_path_base)
        os.system("mkdir -p %s"%output_subdir)
        os.chdir(output_path_base+output_subdir) 
        to_return = os.getcwd()
        #if "output" not in os.listdir("."): os.mkdir("output"); os.mkdir("status")
        #os.system("cp %s/runAnalyzer.sh ./output/"%output_path_base) ; os.system("cp %s/%s ./output/"%(output_path_base,script))
        os.chdir(cwd)
        return to_return

def run_jobs(input_path,output_subdir,processname,year):
        #submit_dir = make_needed_output_dirs(output_subdir)
        submit_dir = cwd + '/farmoutjobs_%s/'%date_string + output_subdir
        output_root_files_dir = output_path_base + 'farmoutjobs_%s/'%date_string + output_subdir
        isdata = False; era =None # not submitting data using this function
        if ("data_samples" in output_subdir): isdata = True
        if(processname=="GJets"): num_files_per_job = 100
        else: num_files_per_job =1
        commandList = ['farmoutAnalysisJobs','--fwklite','--submit-dir=%s'%submit_dir,'--input-dir=%s'%input_path,'--output-dir=%s'%output_root_files_dir,'--max-usercode-size=300','--extra-usercode-files="src/PhysicsTools"','--extra-inputs=%s/keep_branches_*.txt,RootFiles'%cwd,'--input-files-per-job=%s'%num_files_per_job,'--job-generates-output-name','--base-requirements="TARGET.HAS_CMS_HDFS"','--memory-requirements=1000','--disk-requirements=1000','trial1',cmssw_path,script,'--','--isdata=%s'%isdata,'--era=%s'%era,'--processname=%s'%processname,'--year=%s'%year] #'--'seperates options for the script from the submission options
        submit_command  = ' '.join(commandList)
        os.system(submit_command)

def run_jobs_dbs(dataset_name,output_subdir,processname,year,to_get_index=1):# use ,to_get_index=2 for data and ,to_get_index=1 for bkg because structure of dataset_name
        #submit_dir = make_needed_output_dirs(output_subdir)
        print("dataset_name=%s"%dataset_name)
        print("output_subdir=%s"%output_subdir)
        submit_dir = cwd + '/farmoutjobs_%s/'%date_string + output_subdir
        output_root_files_dir = output_path_base + 'farmoutjobs_%s/'%date_string + output_subdir
        isdata = False
        if ("data_samples" in output_subdir): isdata = True
        file_cmd = 'dasgoclient -query="file dataset=%s"'%dataset_name
        filenames = Popen(file_cmd,stdout=PIPE,stderr=STDOUT, shell=True).communicate()[0].split('\n'); filenames=filter(None,filenames)
        to_get = (dataset_name.split('/'))[to_get_index]
        if(isdata): era = to_get.split('-')[0].strip('Run2018')
        else: era = None        
        #'''
        if os.path.isfile('files_%s.in'%to_get): to_get = to_get + (dataset_name.split('/'))[2]# in case of extensions to dataset eg. in case of TTJets
        with open('files_%s.in'%to_get, 'w') as f: 
                for count,item in enumerate(filenames):
                        #if count > 3 : break
                        item = 'root://cmsxrootd.fnal.gov/' + item
                        f.write("%s\n" % item)
        #'''                
        commandList = ['farmoutAnalysisJobs','--fwklite','--submit-dir=%s'%submit_dir,'--assume-input-files-exist','--input-file-list=files_%s.in'%to_get,'--output-dir=%s'%output_root_files_dir,'--max-usercode-size=300','--extra-usercode-files="src/PhysicsTools"','--extra-inputs=%s/keep_branches_*.txt,RootFiles'%cwd,'--input-files-per-job=1','--job-generates-output-name','--base-requirements="TARGET.HAS_CMS_HDFS"','--memory-requirements=1000','--disk-requirements=1000','trial1',cmssw_path,script,'--','--isdata=%s'%isdata,'--era=%s'%era,'--processname=%s'%processname,'--year=%s'%year] #'--'seperates options for the script from the submission options
        submit_command  = ' '.join(commandList)
        os.system(submit_command)

def do_submit():
        for proc_type in os.listdir(input_path_base): # proc_type = ["sig_samples","bkg_samples","data_samples"]
                if(proc_type == "data_samples"): # access directly from dbs
                        #pass
                        #'''
                        data_query_str = '/*MET/Run2018*02Apr2020*/*NANO*' # 2018 SignalRegion and Muon CRs
                        proc = data_query_str.split('/')[1].strip('*')
                        get_cmd = 'dasgoclient -query="dataset=%s"'%data_query_str
                        get_datasets = Popen(get_cmd,stdout=PIPE,stderr=STDOUT, shell=True).communicate()[0].split('\n'); get_datasets = filter(None,get_datasets)
                        print("Submitting jobs for%s"%get_datasets)
                        for dataset in get_datasets:
                                cat = (dataset.split('/'))[2]#.split('-'))[0].strip('Run2018')
                                run_jobs_dbs(dataset,"%s/%s/%s"%(proc_type,proc,cat),"data",year,to_get_index=2)
                        #'''        
                        '''
                        proc_type_path = input_path_base + proc_type + '/'
                        for proc in os.listdir(proc_type_path): #proc=["MET","SingleElectron","SinglePhoton","EGamma"]
                                if proc not in [""]: # to ignore
                                        proc_path = proc_type_path + proc + '/'
                                        for cat in os.listdir(proc_path):  # eg cat for MET2018data = ["A","B","C","D-v1","D-v2"]
                                                cat_path = proc_path + cat + '/'
                                                input_path = cat_path ; run_jobs(input_path,"%s/%s/%s"%(proc_type,proc,cat))             
                        '''                                          
                else: # get from hdfs  
                        #pass
                        #'''
                        proc_type_path = input_path_base + proc_type + '/'
                        for proc in os.listdir(proc_type_path): #proc=["Mx1Mv1000","Z1Jets","Z2Jets","WJets","QCD","DY1Jets","DY2Jets","GJets"]
                                if(1==1):#if proc not in ["GJets"]: # ignoring GJets for now
                                        proc_path = proc_type_path + proc + '/'
                                        if proc_type == "bkg_samples" : # sig_samples have no categories
                                                for cat in os.listdir(proc_path):  # eg cat for z1jets = ["ZPt_50to150","ZPt_150to250","ZPt_250to400","ZPt_400toinf"]
                                                        cat_path = proc_path + cat + '/'
                                                        input_path = cat_path ; run_jobs(input_path,"%s/%s/%s"%(proc_type,proc,cat),proc,year)
                                        else :
                                                input_path =  proc_path ; run_jobs(input_path,"%s/%s"%(proc_type,proc),proc,year)  
                        #'''                         

def do_submit_other_bkgs():
        common_str = '/*Autumn18*v7*/*NANO*'
        other_bkg_dataset_queries = [
                #top datasets
                '/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8',#NLO in QCD
                '/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8',#NLO in QCD
                '/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', #NLO in QCD
                '/ST_tW_top_5f_inclusiveDecays_TuneCP5_*_13TeV-powheg-pythia8', #NLO in QCD
                '/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_*_13TeV-powheg-pythia8', #NLO in QCD
                #diboson
                '/WW_TuneCP5_13TeV-pythia8',# LO in QCD
                '/WZ_TuneCP5_13TeV-pythia8',# LO in QCD
                '/ZZ_TuneCP5_13TeV-pythia8' # LO in QCD
        ]
        other_bkg_dataset_queries = [x+common_str for x in other_bkg_dataset_queries]
        for query_str in other_bkg_dataset_queries:
                proc = query_str.split('/')[1].strip('*')
                if('ST_' in proc): proc = '_'.join(proc.split('_')[:6])
                else: proc = proc.split('_')[0]        
                if(any([x in proc for x in ['WW','WZ','ZZ']])): processname = "diboson"
                else: processname = "top"
                get_cmd = 'dasgoclient -query="dataset=%s"'%query_str
                get_datasets = Popen(get_cmd,stdout=PIPE,stderr=STDOUT, shell=True).communicate()[0].split('\n'); get_datasets = filter(None,get_datasets)
                print("Submitting jobs for%s"%get_datasets)
                for dataset in get_datasets:
                #        cat = (dataset.split('/'))[2]#using this so that same structure can be maintained as other bkg samples
                #        run_jobs_dbs(dataset,"%s/%s/%s"%("bkg_samples",proc,cat),processname,year)
                         run_jobs_dbs(dataset,"%s/%s"%("bkg_samples",proc),processname,year,to_get_index=1)            

# condor test
def do_submit_test():
        #input_path = input_path_base + "sig_samples/Mx1Mv1000/"; processname = "Mx1Mv1000"
        input_path = input_path_base + "bkg_samples/DY2Jets/DY2JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8/"; processname = "DY2Jets"
        output_subdir = cwd + '/farmoutjobs_%s/'%date_string + "farmout_test"
        output_root_files_dir = output_path_base + 'farmoutjobs_%s/'%date_string + "farmout_test"
        isdata = False; era =None
        #submit_dir = make_needed_output_dirs(output_subdir)
        test_command_list = ['farmoutAnalysisJobs','--fwklite','--submit-dir=%s'%output_subdir,'--input-dir=%s'%input_path,'--output-dir=%s'%output_root_files_dir,'--max-usercode-size=300','--extra-usercode-files="src/PhysicsTools"','--extra-inputs=%s/keep_branches_*.txt,RootFiles'%cwd,'--input-files-per-job=1','--job-generates-output-name','--base-requirements="TARGET.HAS_CMS_HDFS"','--memory-requirements=1000','--disk-requirements=1000','trial1',cmssw_path,script,'--','--isdata=%s'%isdata,'--era=%s'%era,'--processname=%s'%processname,'--year=%s'%year] #'--'seperates options for the script from the submission options
        test_command  = ' '.join(test_command_list)
        os.system(test_command) 
        #dbs test
        dataset_name = "/MonoZprime_V_Mx1_Mv1000_TuneCP5_13TeV_madgraph/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM" ;processname = "Mx1Mv1000"
        #dataset_name = "/DY2JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM"; processname = "DY2Jets"
        submit_dir = cwd + '/farmoutjobs_%s/'%date_string + "farmout_test_dbs"
        output_root_files_dir = output_path_base + 'farmoutjobs_%s/'%date_string + "farmout_test_dbs"
        commandList = ['farmoutAnalysisJobs','--fwklite','--submit-dir=%s'%submit_dir,'--assume-input-files-exist','--input-dbs-path=%s'%dataset_name,'--output-dir=%s'%output_root_files_dir,'--max-usercode-size=300','--extra-usercode-files="src/PhysicsTools"','--extra-inputs=%s/keep_branches_*.txt,RootFiles'%cwd,'--input-files-per-job=1','--job-generates-output-name','--base-requirements="TARGET.HAS_CMS_HDFS"','--memory-requirements=1000','--disk-requirements=1000','trial1',cmssw_path,script,'--','--isdata=%s'%isdata,'--era=%s'%era,'--processname=%s'%processname,'--year=%s'%year] #'--'seperates options for the script from the submission options
        submit_command  = ' '.join(commandList)
        os.system(submit_command)

#do_submit_test()
do_submit()
do_submit_other_bkgs()