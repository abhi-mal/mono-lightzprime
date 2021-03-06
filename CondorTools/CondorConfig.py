from collections import OrderedDict
class CondorConfig(OrderedDict):
    def __init__(self,fname=None):
        OrderedDict.__init__(self)
        if fname is None: return
        with open(fname,'r') as f:
            for line in f.readlines():
                pair = line.strip('\n').split('=')
                if len(pair) != 2: continue
                key,value = pair
                key = key.strip()
                if 'Argument' not in key:
                    self[key] = value
                else:
                    if key not in self: self[key] = []
                    self[key].append(value)
    def __str__(self):
        if not any(self): return str(self)
        nspace = max( len(element) for element in self )
        spacer = '{0:<%i}' % nspace
        lines = []
        for key,value in self.iteritems():
            if 'Argument' not in key:
                if type(value) == list: value = ','.join(value)
                lines.append( '%s = %s' % (spacer.format(key),value) )
            else:
                for arg in value: lines.append( '%s = %s\nQueue' % (key,arg) )
        return '\n'.join(lines)
    def write(self,fname):
        with open(fname,'w') as submit: submit.write( str(self) )
                
        

if __name__ == "__main__":

    config = CondorConfig()
    config['x509userproxy'] = '/tmp/x509up_u23272'
    config['universe'] = 'vanilla'
    config['Executable'] = 'runAnalyzer.sh'
    config['label'] = 'MET_4_0'
    config['Batch_Name'] = '$(label)'
    config['Notification'] = 'never'
    config['WhenToTransferOutput'] = 'On_Exit'
    config['ShouldTransferFiles'] = 'yes'
    config['Requirements'] = '(TARGET.UidDomain == "hep.wisc.edu" && TARGET.HAS_CMS_HDFS)'
    config['on_exit_remove'] = '(ExitBySignal == FALSE && (ExitCode == 0 || ExitCode == 42 || NumJobStarts>3))'
    config['+IsFastQueueJob'] = 'True'
    config['getenv'] = 'true'
    config['request_memory'] = 1992
    config['request_disk'] = 2048000
    config['Transfer_Input_Files'] = ['analyze','runAnalyzer.sh']
    config['output'] = '../.status/$(label)/$(Process)_$(label).out'
    config['error']  = '../.status/$(label)/$(Process)_$(label).err'
    config['Log']    = '../.status/$(label)/$(Process)_$(label).log'
    config['Argument'] = ['/path/to/file_$(Process)' for i in range(3)]
    print config
