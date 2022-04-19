#!/usr/bin/env python
import sys
import os
import stat
import math
from subprocess import Popen,STDOUT,PIPE,check_output
from CondorConfig import CondorConfig
from argparse import ArgumentParser

USERPROXY = check_output(["voms-proxy-info","-path"]).split("/")[-1] # Finds User Proxy from command line
DoSubmit = True
ResubmitError = False

def getargs(argv):
    parser = ArgumentParser()
    parser.add_argument('runargs',help='Specify arguments for run',type=str,nargs='+')
    #parser.add_argument('-f','--filelist',help='Use direct filenames as input',action='store_true',default=False)
    #parser.add_argument('--path',help='Specify path to put condor files',default='/nfs_scratch/mallampalli/Analysis/Mono-zprime/all_vars/CMSSW_10_2_18/src/monoJetTools/nanoaod_tests/condor_output/')
    args = parser.parse_args(argv)
    args.error = False
    def checkScript(arg):
        if os.path.isfile(arg): return arg
        else:
            print 'Unable to find script %s' % arg
            args.error = True
    def checkRootDir(arg):
        if os.path.isdir(arg):
            rfiles = [ fn.replace(".root","") for fn in os.listdir(arg) if fn.endswith(".root") ]
            if any(rfiles): return arg,rfiles
            print '%s does not have any root files' % arg
            args.error = True
        else:
            print '%s is not a directory' % arg
            args.error = True
        return None,None
    def checkOutput(arg):
        if arg.endswith('.root'): return arg[:-5]
    def checkSplit(arg):
        nbatches = 1
        nbatches=int(arg.replace("split_",""))
        #If split_-1 is used program will set custom split for each directory so that there are nfile of files in each batch
        if nbatches == -1: nbatches = len(args.rfiles)/NFILE_PER_BATCH
        #Dealing with some edge cases
        if nbatches == 0: nbatches = 1
        return nbatches
    #print("***\nargs.runargs=%s\n***"%args.runargs)
    args.script = checkScript(args.runargs[0])
    args.inputfilelist = args.runargs[1]
    args.outputfile_tag = args.runargs[2] 
    args.label = args.runargs[3]
    args.output_subdir = args.runargs[4]    
    args.input_path,args.rfiles = checkRootDir(args.runargs[5])
    args.output_path_base = args.runargs[6]
    args.isdata = args.runargs[7]
    #checkOutput(args.path+args.output_subdir)
    #args.maxevents = int(args.runargs[3])
    #args.reportevery = int(args.runargs[4])
    #args.nbatches = checkSplit(args.runargs[6])    
    if args.error:
        print 'Errors found in arguments, exitting'
        exit()
    return args

def condor_submit(command,config):
    def DetectedError(config):
        nproc = len(config['Arguments'])
        logdir = os.path.dirname(config['Log']).replace('$(label)',config['label'])
        if not os.path.isdir(logdir): return True
        return nproc != sum( fname.endswith('.log') for fname in os.listdir(logdir) )
    if ResubmitError and DetectedError(config):
        os.system(command)
    elif DoSubmit and not ResubmitError:
        #removeOldFiles(config['outputfile'],config['label'])
        os.system(command)
    else: print "Not Submitting"

def submit(argv=sys.argv,redirect=False):
    args = getargs(argv)
#Beginning to write condor_submit file
    config = CondorConfig() # at this point config is just an ordered dictionary
    config['x509userproxy'] = '/tmp/%s' % USERPROXY
    config['universe'] = 'vanilla'
    config['Executable'] = 'runAnalyzer.sh'
    config['Notification'] = 'never'
    config['WhenToTransferOutput'] = 'On_Exit'
    config['Requirements'] = '(TARGET.UidDomain == "hep.wisc.edu" && TARGET.HAS_CMS_HDFS)'
    config['on_exit_remove'] = '(ExitBySignal == FALSE && (ExitCode == 0 || ExitCode == 42 || NumJobStarts>3))'
    config['+IsFastQueueJob'] = 'True'
    config['getenv'] = 'true'
    config['request_memory'] = 2000
    config['request_disk'] = 2048#2048000
    config['script'] = args.script
    config['inputfilelist'] = args.inputfilelist
    config['outputfile_tag'] = args.outputfile_tag
    #config['maxevents'] = args.maxevents
   # config['reportevery'] = args.reportevery
    config['label'] = args.label
   # config['Batch_Name'] = '%s%s_$(label)' % (args.region,args.year)
    config['Transfer_Input_Files'] = ['$(script)'] #+ [args.inputfilelist] #+ [args.physicstools_path]#[args.input_path]
    #print(args.input_path)
    #print(args.inputfilelist)
    #print("Transferring: %s"%config['Transfer_Input_Files'])
    config['output'] = '../status/$(Process)_$(label).out'
    config['error']  = '../status/$(Process)_$(label).err'
    config['Log']    = '../status/$(Process)_$(label).log'
#    if not args.filelist:    
#        stripDataset(args.rfiles)
#        splitArgument(args.nbatches,args.rfiles,config,redirect)
#    else:
#        inputFilelist(args.nbatches,args.rfiles,config,redirect)
    arguments = []
    arguments.append("$(script) $(outputfile_tag) %s $(inputfilelist)"%args.isdata)
    #arguments.append("$(script) --outputFilename_tag=$(outputfile_tag) --isdata=%s --input_skimfile_list=$(inputfilelist)"%args.isdata)
    config['Arguments'] = arguments

    config.write('%s%s/output/condor_%s' % (args.output_path_base,args.output_subdir,args.label))
    #Move into .output/ and run newly made condor_submit file
    cwd = os.getcwd()
    os.chdir("%s%s/output/"%(args.output_path_base,args.output_subdir))
    command = "condor_submit condor_%s" % args.label
    if redirect is not False:
        redirect.close()
        condor_submit(command + ' >> ../status/%s/submit.txt' % args.label,config)
    else:
        condor_submit(command,config)
    os.chdir(cwd)


