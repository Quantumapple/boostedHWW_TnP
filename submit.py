import json
import argparse
import os
from jinja2 import Template
from pathlib import Path

def make_job(args):
    with open(f'{args.input_json}', 'r') as inputfile:
        fileset = json.load(inputfile)

    json_dir = current_dir / 'splited_json'
    json_dir.mkdir(exist_ok=True)

    if json_dir.exists() and json_dir.is_dir():
        for item in json_dir.iterdir():
            if item.is_file():
                item.unlink()

    n_split = args.split
    for ikey, flist in fileset.items():
        loops = len(flist)//n_split
        for iloop in range(loops+1):
            updated_filelist = {}
            updated_filelist[ikey] = ["root://cmseos.fnal.gov/" + f for f in flist[n_split*iloop:n_split*(iloop+1)]]

            with open(f'{json_dir}/{ikey}_{iloop}.json', 'w') as outfile:
                json.dump(updated_filelist, outfile, indent=4)

    files = Path(f'{json_dir}').glob('*json')
    listfile = current_dir / 'input_list_for_condor.txt'
    if listfile.is_file():
        listfile.unlink()

    with open(listfile, 'a') as listfile:
        for ifile in files:
            name = ifile.name
            save_string = f"{name}, {name.split('.')[0]}"
            listfile.write(save_string + '\n')

    #### Make python command
    bash_command = "python run.py --year {{ year }} --channel {{ channel }} --executor {{ executor }} --input_json {{ input_json }} --output {{ output }}"

    options = {
        'year': args.year,
        'channel': args.channel,
        'executor': args.executor,
        'input_json': '${1}',
        'output': '${2}',
    }

    conditional_args = {
        'same_sign_bkg': args.same_sign_bkg,
        'full_mass_range_bkg': args.full_mass_range_bkg,
    }

    for arg, value in conditional_args.items():
        if value:
            bash_command += f" --{arg}"  # Add the argument if value is True

    # Define the bash script template
    bash_template = """#!/bin/bash

mv splited_json/{1} ./
ls -ltrh
echo ""
pwd

echo "{0}"
{0}

echo ""
ls -ltrh
echo ""

rm *.json
""".format(bash_command, '${1}')

    bash_script = Template(bash_template).render(options)
    with open('run_TnP.sh','w') as bashfile:
        bashfile.write(bash_script)

    outdir = current_dir / 'histograms'
    outdir.mkdir(exist_ok=True)
    jdl = """universe              = vanilla
executable            = run_TnP.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(ifile) $(name)
transfer_Input_Files  = hww_muon_TnP.py,run.py,{2}
TransferOutputRemaps = "$(name).sqlite={1}/$(name).sqlite"
output                = {0}/$(ClusterId).$(ProcId).stdout
error                 = {0}/$(ClusterId).$(ProcId).stderr
log                   = {0}/$(ClusterId).$(ProcId).log
+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base:0.7.21-py3.10"
Queue ifile,name from input_list_for_condor.txt
""".format(str(log_dir), str(outdir), str(json_dir))

    with open(f'condor_TnP.jdl','w') as jdlfile:
        jdlfile.write(jdl)

    if args.dryrun:
        print('=========== Input text file ===========')
        os.system('cat input_list_for_condor.txt')
        print()
        print('=========== Bash file ===========')
        os.system('cat run_TnP.sh')
        print()
        print()
        print('=========== JDL file ===========')
        os.system('cat condor_TnP.jdl')
    else:
        os.system(f'condor_submit condor_TnP.jdl')


####################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--year", dest="year", default="2018", help="year", type=str)
    parser.add_argument("--channel", dest="channel", default="muon", help="Choose lepton, either muon or electron", type=str)
    parser.add_argument("--executor", dest="executor", default="futures", help="Coffea Executor. dask, futures", type=str)
    parser.add_argument("--input_json", dest="input_json", help="Input json file", type=str)
    parser.add_argument("--split", dest="split", default=30, help="number to split files per a single job", type=int)
    parser.add_argument("--same_sign_bkg", dest="same_sign_bkg", action="store_true")
    parser.add_argument("--full_mass_range_bkg", dest="full_mass_range_bkg", action="store_true")
    parser.add_argument("--dryrun", dest="dryrun", action="store_true")

    args = parser.parse_args()

    current_dir = Path('./')
    log_dir = current_dir / 'condor_logs'
    log_dir.mkdir(exist_ok=True)

    if log_dir.exists():
        os.system('rm condor_logs/*log')
        os.system('rm condor_logs/*stdout')
        os.system('rm condor_logs/*stderr')
        os.system('ls condor_logs/*log | wc -l')

    make_job(args)
