from coffea.nanoevents import PFNanoAODSchema
from hww_muon_TnP import MuonTnpProcessor
from coffea import processor

import uproot
import json, time
import argparse

def run_processor(args):

    with open(f'{args.input_json}', 'r') as inputfile:
        fileset = json.load(inputfile)
        files_to_use = fileset

    if args.local:
        files_to_use = {}
        for ikey, flist in fileset.items():
            files_to_use[ikey] = ["root://cmseos.fnal.gov/" + f for f in flist]

    tic = time.time()
    if args.executor == "dask":
        pass
        # from coffea.nanoevents import NanoeventsSchemaPlugin
        # from distributed import Client
        # from lpcjobqueue import LPCCondorCluster

        # cluster = LPCCondorCluster(
        #     ship_env=True,
        #     transfer_input_files="boostedhiggs",
        # )
        # client = Client(cluster)
        # nanoevents_plugin = NanoeventsSchemaPlugin()
        # client.register_worker_plugin(nanoevents_plugin)
        # cluster.adapt(minimum=1, maximum=30)

        # print("Waiting for at least one worker")
        # client.wait_for_workers(1)

        # # does treereduction help?
        # executor = processor.DaskExecutor(status=True, client=client, treereduction=2)

    else:
        uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

        if args.executor == "futures":
            executor = processor.FuturesExecutor(status=True)
        else:
            executor = processor.IterativeExecutor(status=True)

    proc = MuonTnpProcessor(year=args.year, channels=args.channel, sqlite_output_name=args.output, same_sign_bkg=args.same_sign_bkg, full_mass_bkg=args.full_mass_range_bkg)

    run = processor.Runner(
        executor=executor,
        savemetrics=True,
        schema=PFNanoAODSchema,
        skipbadfiles = True,
    )
    out, metrics = run(files_to_use, "Events", processor_instance=proc)

    elapsed = time.time() - tic
    print(f"Metrics: {metrics}")
    print(f"Finished in {elapsed:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--year", dest="year", default="2018", help="year", type=str)
    parser.add_argument("--channel", dest="channel", default="muon", help="Choose lepton, either muon or electron", type=str)
    parser.add_argument("--executor", dest="executor", default="futures", help="Coffea Executor. dask, futures", type=str)
    parser.add_argument("--input_json", dest="input_json", help="Input json file", type=str)
    parser.add_argument("--output", dest="output", default="outputs", help="Output file name", type=str)
    parser.add_argument("--same_sign_bkg", dest="same_sign_bkg", action="store_true")
    parser.add_argument("--full_mass_range_bkg", dest="full_mass_range_bkg", action="store_true")
    parser.add_argument("--local", dest="local", action="store_true")

    args = parser.parse_args()

    run_processor(args)
