from coffea import processor
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import Weights
import awkward as ak
import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from corrections import (
    add_lepton_weight,
    add_pileup_weight,
    add_pileupid_weights,
    add_VJets_kFactors,
)

### Select events based on HLT and number of muon = 2 in the event
def preSelect_Events(
    events,
    year: str,
    which_lepton: str,
    isData: bool,
):
    """Preselect events by HLT trigger, ID, and number of leptons. ID selection is fixed to loose selection.
    events: NanoEventsFactory,
        Events parsing to Coffea Processor.
    year: str,
        Year.
    which_lepton: str,
        Select muon or electron to do TnP measurement.
    isData: bool,
        Flag to apply LumiMask on data
    """

    hlt_triggers = {
        "2016postVFP": {
            "muon": ["Mu50", "TkMu50", "IsoMu24", "IsoTkMu24"],
            "electron": ["Ele27_WPTight_Gsf", "Ele115_CaloIdVT_GsfTrkIdT", "Photon175"],
        },
        "2016preVFP": {
            "muon": ["Mu50", "TkMu50", "IsoMu24", "IsoTkMu24"],
            "electron": ["Ele27_WPTight_Gsf", "Ele115_CaloIdVT_GsfTrkIdT", "Photon175"],
        },
        "2017": {
            "muon": ["Mu50", "IsoMu27", "OldMu100", "TkMu100"],
            "electron": ["Ele35_WPTight_Gsf", "Ele115_CaloIdVT_GsfTrkIdT", "Photon200"],
        },
        "2018": {
            "muon": ["Mu50", "IsoMu24", "OldMu100", "TkMu100"],
            "electron": ["Ele32_WPTight_Gsf", "Ele115_CaloIdVT_GsfTrkIdT", "Photon200"],
        }
    }

    met_filters_dict = {
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        '2016postVFP': [
                'goodVertices',
                'globalSuperTightHalo2016Filter',
                'HBHENoiseFilter',
                'HBHENoiseIsoFilter',
                'EcalDeadCellTriggerPrimitiveFilter',
                'BadPFMuonFilter',
                'BadPFMuonDzFilter',
                'eeBadScFilter'
                ],

        '2016preVFP': [
                'goodVertices',
                'globalSuperTightHalo2016Filter',
                'HBHENoiseFilter',
                'HBHENoiseIsoFilter',
                'EcalDeadCellTriggerPrimitiveFilter',
                'BadPFMuonFilter',
                'BadPFMuonDzFilter',
                'eeBadScFilter'
                ],

        '2017': [
                'goodVertices',
                'globalSuperTightHalo2016Filter',
                'HBHENoiseFilter',
                'HBHENoiseIsoFilter',
                'EcalDeadCellTriggerPrimitiveFilter',
                'BadPFMuonFilter',
                'BadPFMuonDzFilter',
                'eeBadScFilter',
                'ecalBadCalibFilter'
                ],

        '2018': [
                'goodVertices',
                'globalSuperTightHalo2016Filter',
                'HBHENoiseFilter',
                'HBHENoiseIsoFilter',
                'EcalDeadCellTriggerPrimitiveFilter',
                'BadPFMuonFilter',
                'BadPFMuonDzFilter',
                'eeBadScFilter',
                'ecalBadCalibFilter'
                ]
    }

    lumiMasks = {
        '2016preVFP': LumiMask("utils/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
        '2016postVFP': LumiMask("utils/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
        '2017': LumiMask("utils/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"),
        '2018': LumiMask("utils/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"),
    }

    nevents = len(events)

    # HLT trigger selection, logical OR
    pass_hlt = np.zeros(nevents, dtype="bool")
    for t in hlt_triggers[year][which_lepton]:
        if t in events.HLT.fields:
            pass_hlt = pass_hlt | events.HLT[t]

    # MET filters
    met_filters =  np.ones(nevents, dtype='bool')
    for flag in met_filters_dict[year]:
        met_filters = met_filters & events.Flag[flag]

    # Luminosity mask based on Golden Json
    lumimask = np.ones(nevents, dtype='bool')
    if isData:
        lumimask = lumiMasks[year](events.run, events.luminosityBlock)

    good_loc = pass_hlt & met_filters & lumimask
    good_events = events[good_loc]
    return good_events, good_loc

def _trigger_match(tag_lepton, trigobjs, lepton_id, pt, filterbit):
    pass_pt = trigobjs.pt > pt
    pass_id = abs(trigobjs.id) == lepton_id
    pass_filterbit = trigobjs.filterBits & (0x1 << filterbit) > 0
    trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
    delta_r = tag_lepton.metric_table(trigger_cands)
    pass_delta_r = delta_r < 0.1
    n_of_trigger_matches = ak.sum(pass_delta_r, axis=2)
    trig_matched_locs = n_of_trigger_matches >= 1
    return trig_matched_locs

def _process_zcands_for_probes(
        zcands,
        good_events,
        pt_tags,
        pt_probes,
        abseta_tags,
        abseta_probes,
        same_sign_bkg: bool = False,
    ):
    ## pT, eta, looseID selections
    pt_cond_tags = zcands.tag.pt > pt_tags
    eta_cond_tags = abs(zcands.tag.eta) < abseta_tags
    pt_cond_probes = zcands.probe.pt > pt_probes
    eta_cond_probes = abs(zcands.probe.eta) < abseta_probes
    loose_id_for_tag = zcands.tag.looseId
    loose_id_for_probe = zcands.probe.looseId
    zcands = zcands[pt_cond_tags & pt_cond_probes & eta_cond_tags & eta_cond_probes & loose_id_for_tag & loose_id_for_probe]

    ## Mass checking
    mass = (zcands.tag + zcands.probe).mass
    opposite_charge = zcands.tag.charge * zcands.probe.charge == -1
    # in_mass_window = (mass > 0)
    in_mass_window = (mass > 60) & (mass < 120)

    if same_sign_bkg:
        isZ = in_mass_window & ~opposite_charge
    else:
        isZ = in_mass_window & opposite_charge

    zcands = zcands[isZ]

    ## delta R between tag and probe, make sure they are not the same
    dr = zcands.tag.delta_r(zcands.probe)
    dr_condition = dr > 0.0
    zcands = zcands[dr_condition]

    ## Select the event when lepton pair
    has_pair = ak.num(zcands) >= 1
    zcands = zcands[has_pair]
    good_events = good_events[has_pair]

    return zcands, good_events

def _process_zcands_for_tags(
        zcands,
        good_events,
        lepton_id,
        trigger_obj_pt,
        filter_bit,
    ):

    ## tag object matching with triggr object and tight ID requirement
    trigobjs = good_events.TrigObj
    trig_matched_tag = _trigger_match(zcands.tag, trigobjs, lepton_id, trigger_obj_pt, filter_bit)
    tight_id_for_tag = zcands.tag.tightId
    zcands = zcands[trig_matched_tag&tight_id_for_tag]

    ## Tag should exist in a pair
    events_with_tags = ak.num(zcands.tag, axis=1) >= 1
    zcands = zcands[events_with_tags]
    good_events = good_events[events_with_tags]
    has_pair = ak.num(zcands) >= 1

    ## Return final objects
    zcands = zcands[has_pair]
    all_probe_events = good_events[has_pair]
    all_probe_events["probe_lep"] = zcands.probe
    all_probe_events["tag_lep"] = zcands.tag
    all_probe_events["pair_mass"] = (all_probe_events["probe_lep"] + all_probe_events["tag_lep"]).mass

    return all_probe_events, has_pair

### Select the combination which mass is the closest to the Z boson mass
def find_indices_same_shape(a, b):
    result = []
    for sublist, b_elem in zip(a, b):
        indices = [i for i, val in enumerate(sublist) if np.equal(val, b_elem)]
        result.append(indices)
    return result

def find_TnP_closest_to_Z(good_TnP_events):
    ### This is necessary, if we want to select TnP pairs closest to Z boson mass
    tmp1 = abs(good_TnP_events.pair_mass-91.1876)
    tmp2 = ak.min(abs(good_TnP_events.pair_mass-91.1876), axis=1)
    selected_args = ak.Array(find_indices_same_shape(tmp1, tmp2))
    probe_leps = good_TnP_events.probe_lep[selected_args]
    tag_leps = good_TnP_events.tag_lep[selected_args]

    return tag_leps, probe_leps

def find_minimum_dR(lepton, fatjet):

    if len(lepton) != len(fatjet):
        raise ValueError('Lepton and fatjet length are not matched')

    ### Find minimum delta R with probe
    lep, fj = ak.unzip(ak.cartesian([lepton, fatjet]))
    fj_idx_lep = ak.argmin(lep.delta_r(fj), axis=1, keepdims=True)
    selected_fj = fj[fj_idx_lep]

    ### Again, recalculate delta R using the AK8 jet closest to the probe lepton
    lep, fj = ak.unzip(ak.cartesian([lepton, selected_fj]))

    return lep, fj

def _hww_muon_selections(
        muons,
    ):

    pt = (muons.pt > 30)
    eta = (np.abs(muons.eta) < 2.4)
    id_selection = muons.mediumId

    #isolation_cut = 0.2
    isolation = (((muons.pfRelIso04_all < 0.2) & (muons.pt < 55)) | (muons.pt >= 55) & (muons.miniPFRelIso_all < 0.2))
    vertex = (np.abs(muons.dz) < 0.1) & (np.abs(muons.dxy) < 0.02)
    loc = pt & eta & id_selection & isolation & vertex

    return loc

def merge_to_pandas_df(base_df, array_to_merge, column_title):

    temp_df = ak.to_pandas(array_to_merge).reset_index()
    temp_df.columns = ['entry', column_title]

    merged_df = base_df.merge(temp_df, left_on='evt', right_on='entry', how='left')
    merged_df.drop(columns='entry', inplace=True)
    del temp_df

    return merged_df

class MuonTnpProcessor(processor.ProcessorABC):
    def __init__(self, year='2018', channels='muon', sqlite_output_name='output', met_cut=None):
        self._year = year
        self._channels = channels
        self._sqlite_output_name = sqlite_output_name
        self._met_cut = met_cut

    def process(self, events):
        dataset = events.metadata['dataset']
        output = {}

        isData = not hasattr(events, "genWeight")
        # if not isData:
        #     output['sumw'] = ak.sum(events.genWeight)

        ### Preselection - HLT trigger
        good_events, _ = preSelect_Events(events, self._year, self._channels, isData)

        ### double, triple (and more) the number of TnP candidates (each item in the pair can be both a tag and a probe)
        ij = ak.argcartesian([good_events.Muon, good_events.Muon])
        is_not_diag = ij["0"] != ij["1"]
        i, j = ak.unzip(ij[is_not_diag])
        zcands = ak.zip({"tag": good_events.Muon[i], "probe": good_events.Muon[j]})

        ### Minimum Muon pT cut > 30GeV, pleatau after trigger
        ### Some minimal pt cut for probe is required in the future!!
        ### ID identifcation shouldn't be depend on pT, so apply the same pT cut for both leptons.
        ### Do not convolute pT dependence

        ### Method 1: Select pairs with the same sign
        ss_pairs, ss_events_pre = _process_zcands_for_probes(
            zcands=zcands,
            good_events=good_events,
            pt_tags=30.,
            pt_probes=30.,
            abseta_tags=2.4,
            abseta_probes=2.4,
            same_sign_bkg = True,
        )

        ### Select good TnP events
        ss_events, ss_loc = _process_zcands_for_tags(
            zcands = ss_pairs,
            good_events = ss_events_pre,
            lepton_id = 13,
            trigger_obj_pt = 30.,
            filter_bit = 1,
        )

        ss_tag_leps, ss_probe_leps = ss_events.tag_lep, ss_events.probe_lep
        ### This is necessary, if we want to select TnP pairs closest to Z boson mass
        # ss_tag_leps, ss_probe_leps = find_TnP_closest_to_Z(ss_events)

        ### Method 2: use full mass range, control by input argument
        ### Select good probe events, pairs
        good_zcands, good_zcands_events = _process_zcands_for_probes(
            zcands=zcands,
            good_events=good_events,
            pt_tags=30.,
            pt_probes=30.,
            abseta_tags=2.4,
            abseta_probes=2.4,
        )

        ### Select good TnP events
        good_TnP_events, good_TnP_loc = _process_zcands_for_tags(
            zcands = good_zcands,
            good_events = good_zcands_events,
            lepton_id = 13,
            trigger_obj_pt = 30.,
            filter_bit = 1,
        )

        tag_leps, probe_leps = good_TnP_events.tag_lep, good_TnP_events.probe_lep
        ### This is necessary, if we want to select TnP pairs closest to Z boson mass
        # tag_leps, probe_leps = find_TnP_closest_to_Z(good_TnP_events)
        # if not self._full_mass_bkg:
        #     tag_leps, probe_leps = find_TnP_closest_to_Z(good_TnP_events)
        # else:
        #     probe_leps = good_TnP_events.probe_lep
        #     tag_leps = good_TnP_events.tag_lep

        ### AK4 jets for PU ID weights and HEM cleaning
        jets = good_TnP_events.Jet
        jet_selector = (
            (jets.pt > 30)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
            & ((jets.pt >= 50) | ((jets.pt < 50) & (jets.puId & 2) == 2))
        )
        jets = jets[jet_selector]

        ### Electrons for HEM cleaning
        electrons = good_TnP_events.Electron

        # hem-cleaning selection
        if self._year == "2018":
            hem_veto = ak.any(
                ((jets.eta > -3.2) & (jets.eta < -1.3) & (jets.phi > -1.57) & (jets.phi < -0.87)),
                -1,
            ) | ak.any(
                (
                    (electrons.pt > 30)
                    & (electrons.eta > -3.2)
                    & (electrons.eta < -1.3)
                    & (electrons.phi > -1.57)
                    & (electrons.phi < -0.87)
                ),
                -1,
            )

            hem_cleaning = (
                ((good_TnP_events.run >= 319077) & (isData))  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(good_TnP_events)) < 0.632) & (not isData))
            ) & (hem_veto)


        pass_loc = _hww_muon_selections(probe_leps)
        z_mass = (tag_leps+probe_leps).mass

        df = pd.concat([ak.to_pandas(z_mass), ak.to_pandas(tag_leps.pt), ak.to_pandas(probe_leps.pt), ak.to_pandas(probe_leps.eta), ak.to_pandas(pass_loc)], axis=1)
        df.reset_index(inplace=True)
        df.columns = ['evt', 'identifier', 'mass', 'tag_pt', 'probe_pt', 'probe_eta', 'pass_hww']

        step1_df = merge_to_pandas_df(df, hem_cleaning, 'isHEM')
        del df

        ### MET
        met = good_TnP_events.MET
        step2_df = merge_to_pandas_df(step1_df, met.pt, 'met_pt')
        del step1_df, met

        ### Calculate delta R with AK8 fatjet for both tag and probe leptons
        ### Then use the fatjet with minimal Delta R with probe lepton
        ### See the reference: https://github.com/jieunyoo/boostedhiggs_may27/blob/main/boostedhiggs/vhprocessor.py#L298-L300
        fatjet = good_TnP_events.FatJet
        fatjet_selector = (fatjet.pt > 200) & (abs(fatjet.eta) < 2.5) & fatjet.isTight
        fatjet = fatjet[fatjet_selector]
        has_good_fj = ak.num(fatjet) >= 1

        merged_df = merge_to_pandas_df(step2_df, has_good_fj, 'has_fj')
        merged_df = merge_to_pandas_df(merged_df, ak.num(fatjet), 'nfj')
        del step2_df

        ### At least one good Ak8 jet
        # fj_correlated_tag_leps = tag_leps[has_good_fj]
        fj_correlated_probe_leps = probe_leps[has_good_fj]

        min_lep, min_fj = find_minimum_dR(fj_correlated_probe_leps, fatjet[has_good_fj])
        min_dr = ak.to_pandas(min_lep.delta_r(min_fj)).reset_index()['values'].values

        filtered_df = merged_df.loc[merged_df['has_fj']]
        filtered_df.loc[:, 'min_dr'] = min_dr
        merged_df.loc[merged_df['has_fj'], 'min_dr'] = filtered_df['min_dr']
        del filtered_df, min_dr

        if self._met_cut is not None:
            merged_df = merged_df.loc[merged_df['met_pt'] < self._met_cut]

        outfile = f'{self._sqlite_output_name}.sqlite'
        with sqlite3.connect(outfile) as sqlconn:
            merged_df.to_sql('signal', sqlconn, if_exists='append', index=False)

        #################
        ### Same sign bkg
        #################
        pass_loc = _hww_muon_selections(ss_probe_leps)
        not_z_mass = (ss_tag_leps+ss_probe_leps).mass

        ss_df = pd.concat([ak.to_pandas(not_z_mass), ak.to_pandas(ss_tag_leps.pt), ak.to_pandas(ss_probe_leps.pt), ak.to_pandas(ss_probe_leps.eta), ak.to_pandas(pass_loc)], axis=1)
        ss_df.reset_index(inplace=True)
        ss_df.columns = ['evt', 'identifier', 'mass', 'tag_pt', 'probe_pt', 'probe_eta', 'pass_hww']

        ### AK4 jets for PU ID weights and HEM cleaning
        jets = ss_events.Jet
        jet_selector = (
            (jets.pt > 30)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
            & ((jets.pt >= 50) | ((jets.pt < 50) & (jets.puId & 2) == 2))
        )
        jets = jets[jet_selector]

        ### Electrons for HEM cleaning
        electrons = ss_events.Electron

        # hem-cleaning selection
        if self._year == "2018":
            hem_veto = ak.any(
                ((jets.eta > -3.2) & (jets.eta < -1.3) & (jets.phi > -1.57) & (jets.phi < -0.87)),
                -1,
            ) | ak.any(
                (
                    (electrons.pt > 30)
                    & (electrons.eta > -3.2)
                    & (electrons.eta < -1.3)
                    & (electrons.phi > -1.57)
                    & (electrons.phi < -0.87)
                ),
                -1,
            )

            hem_cleaning = (
                ((ss_events.run >= 319077) & (isData))  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(ss_events)) < 0.632) & (not isData))
                ) & (hem_veto)

        ss_step1_df = merge_to_pandas_df(ss_df, hem_cleaning, 'isHEM')
        del ss_df

        ### MET
        met = good_events.MET[ss_loc]
        ss_step2_df = merge_to_pandas_df(ss_step1_df, met.pt, 'met_pt')
        del ss_step1_df, met

        fatjet = good_events.FatJet[ss_loc]
        fatjet_selector = (fatjet.pt > 200) & (abs(fatjet.eta) < 2.5) & fatjet.isTight
        fatjet = fatjet[fatjet_selector]
        has_good_fj = ak.num(fatjet) >= 1

        merged_ss_df = merge_to_pandas_df(ss_step2_df, has_good_fj, 'has_fj')
        merged_ss_df = merge_to_pandas_df(merged_ss_df, ak.num(fatjet), 'nfj')
        del ss_step2_df

        # fj_corr_ss_tag_leps = ss_tag_leps[has_good_fj]
        fj_corr_ss_probe_leps = ss_probe_leps[has_good_fj]

        min_lep, min_fj = find_minimum_dR(fj_corr_ss_probe_leps, fatjet[has_good_fj])
        min_dr = ak.to_pandas(min_lep.delta_r(min_fj)).reset_index()['values'].values

        filtered_df = merged_ss_df.loc[merged_ss_df['has_fj']]
        filtered_df.loc[:, 'min_dr'] = min_dr
        merged_ss_df.loc[merged_ss_df['has_fj'], 'min_dr'] = filtered_df['min_dr']
        del filtered_df, min_dr

        if self._met_cut is not None:
            merged_ss_df = merged_ss_df.loc[merged_ss_df['met_pt'] < self._met_cut]

        with sqlite3.connect(outfile) as sqlconn:
            merged_ss_df.to_sql('same_sign_bkg', sqlconn, if_exists='append', index=False)


        ###########
        ### Weights
        ###########
        # if not isData:
        #     weights = Weights(len(good_TnP_events), storeIndividual=True)
        #     weights.add("genweight", good_TnP_events.genWeight)

        #     if self._year in ("2016preVFP", "2016postVFP", "2017"):
        #         weights.add("L1Prefiring", good_TnP_events.L1PreFiringWeight.Nom, good_TnP_events.L1PreFiringWeight.Up, good_TnP_events.L1PreFiringWeight.Dn)

        #     add_pileup_weight(weights, self._year, nPU=ak.to_numpy(good_TnP_events.Pileup.nPU))
        #     add_pileupid_weights(weights, jets, good_TnP_events.GenJet, wp="L")
        #     add_lepton_weight(weights, probe_leps, self._year, "muon")

        #     ewk_corr, qcd_corr, alt_qcd_corr = add_VJets_kFactors(weights, good_TnP_events.GenPart, dataset, good_TnP_events)
        #     # # add corrections for plotting
        #     # variables["weight_ewkcorr"] = ewk_corr
        #     # variables["weight_qcdcorr"] = qcd_corr
        #     # variables["weight_altqcdcorr"] = alt_qcd_corr

        #     print(ak.to_pandas(weights.weight()))


        return output

    def postprocess(self, accumulator):
        return accumulator
