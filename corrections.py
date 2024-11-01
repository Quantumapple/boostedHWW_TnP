import importlib.resources
import awkward as ak
import correctionlib
import numpy as np
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods.nanoaod import JetArray

ak.behavior.update(vector.behavior)

# with importlib.resources.path("boostedhiggs.data", "msdcorr.json") as filename:
#     msdcorr = correctionlib.CorrectionSet.from_file(str(filename))

# def corrected_msoftdrop(fatjets):
#     msdraw = np.sqrt(
#         np.maximum(
#             0.0,
#             (fatjets.subjets * (1 - fatjets.subjets.rawFactor)).sum().mass2,
#         )
#     )
#     # msoftdrop = fatjets.msoftdrop
#     msdfjcorr = msdraw / (1 - fatjets.rawFactor)

#     corr = msdcorr["msdfjcorr"].evaluate(
#         np.array(ak.flatten(msdfjcorr / fatjets.pt)),
#         np.array(ak.flatten(np.log(fatjets.pt))),
#         np.array(ak.flatten(fatjets.eta)),
#     )
#     corr = ak.unflatten(corr, ak.num(fatjets))
#     corrected_mass = msdfjcorr * corr

#     return corrected_mass

with importlib.resources.path("utils", "ULvjets_corrections.json") as filename:
    vjets_kfactors = correctionlib.CorrectionSet.from_file(str(filename))

def get_vpt(genpart, check_offshell=False):
    """Only the leptonic samples have no resonance in the decay tree, and only
    when M is beyond the configured Breit-Wigner cutoff (usually 15*width)
    """
    boson = ak.firsts(
        genpart[((genpart.pdgId == 23) | (abs(genpart.pdgId) == 24)) & genpart.hasFlags(["fromHardProcess", "isLastCopy"])]
    )
    if check_offshell:
        offshell = genpart[
            genpart.hasFlags(["fromHardProcess", "isLastCopy"])
            & ak.is_none(boson)
            & (abs(genpart.pdgId) >= 11)
            & (abs(genpart.pdgId) <= 16)
        ].sum()
        return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
    return np.array(ak.fill_none(boson.pt, 0.0))


def add_VJets_kFactors(weights, genpart, dataset, events):
    """Revised version of add_VJets_NLOkFactor, for both NLO EW and ~NNLO QCD"""

    common_systs = [
        "d1K_NLO",
        "d2K_NLO",
        "d3K_NLO",
        "d1kappa_EW",
    ]
    zsysts = common_systs + [
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    znlosysts = [
        "d1kappa_EW",
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    wsysts = common_systs + [
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]
    wnlosysts = [
        "d1kappa_EW",
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]

    def add_systs(systlist, qcdcorr, ewkcorr, vpt):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom)
        ones = np.ones_like(vpt)
        for syst in systlist:
            weights.add(
                syst,
                ones,
                ewkcorr.evaluate(syst + "_up", vpt) / ewknom,
                ewkcorr.evaluate(syst + "_down", vpt) / ewknom,
            )

    vpt = get_vpt(genpart)
    qcdcorr = np.ones_like(vpt)
    ewcorr = np.ones_like(vpt)

    # alternative QCD NLO correction (for WJets)
    # derived from https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2019/229
    alt_qcdcorr = np.ones_like(vpt)

    if "ZJetsToQQ_HT" in dataset or "DYJetsToLL_M-" in dataset:
        qcdcorr = vjets_kfactors["ULZ_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        ewcorr = ewkcorr.evaluate("nominal", vpt)
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)

    elif "DYJetsToLL_Pt" in dataset or "DYJetsToLL_LHEFilterPtZ" in dataset:
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        ewcorr = ewkcorr.evaluate("nominal", vpt)
        add_systs(znlosysts, qcdcorr, ewkcorr, vpt)

    elif "WJetsToLNu_1J" in dataset or "WJetsToLNu_0J" in dataset or "WJetsToLNu_2J" in dataset:
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        ewcorr = ewkcorr.evaluate("nominal", vpt)
        add_systs(wnlosysts, qcdcorr, ewkcorr, vpt)

    elif "WJetsToQQ_HT" in dataset or "WJetsToLNu_HT" in dataset or "WJetsToLNu_TuneCP5" in dataset:
        qcdcorr = vjets_kfactors["ULW_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        ewcorr = ewkcorr.evaluate("nominal", vpt)
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)

        # added by farouk
        """
        from: https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2019/229
        Bhadrons       Systematic
        0             1.628±0.005 - (1.339±0.020)·10−3 pT(V)
        1             1.586±0.027 - (1.531±0.112)·10−3 pT(V)
        2             1.440±0.048 - (0.925±0.203)·10−3 pT(V)
        """
        genjets = events.GenJet
        goodgenjets = genjets[(genjets.pt > 20.0) & (np.abs(genjets.eta) < 2.4)]

        nB0 = (ak.sum(goodgenjets.hadronFlavour == 5, axis=1) == 0).to_numpy()
        nB1 = (ak.sum(goodgenjets.hadronFlavour == 5, axis=1) == 1).to_numpy()
        nB2 = (ak.sum(goodgenjets.hadronFlavour == 5, axis=1) == 2).to_numpy()

        alt_qcdcorr[nB0] = 1.628 - (1.339 * 1e-3 * vpt[nB0])
        alt_qcdcorr[nB1] = 1.586 - (1.531 * 1e-3 * vpt[nB1])
        alt_qcdcorr[nB2] = 1.440 - (0.925 * 1e-3 * vpt[nB2])

    return ewcorr, qcdcorr, alt_qcdcorr

"""
Lepton Scale Factors
----

Muons:
https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2016
https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017
https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018

- UL CorrectionLib html files:
  https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/MUO_2017_UL_muon_Z.html
  e.g. one example of the correction json files can be found here:
  https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/raw/master/Run2/UL/2017/2017_trigger/Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers_schemaV2.json
  - Trigger iso and non-iso
  - Isolation: We use RelIso<0.25 (LooseRelIso) with medium prompt ID
  - Reconstruction ID: We use medium prompt ID

Electrons:
- UL CorrectionLib htmlfiles:
  https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
  - ID and Isolation:
    - wp90noiso for high pT electrons
    - wp90iso for low pT electrons
  - Reconstruction: RecoAbove20
  - Trigger: Derived using EGamma recommendation: https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTScaleFactorMeasurements
"""

lepton_corrections = {
    "trigger_iso": {
        "muon": {  # For IsoMu24 (| IsoTkMu24 )
            "2016preVFP": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # preVBP
            "2016postVFP": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # postVBF
            "2017": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "2018": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
        },
    },
    "trigger_noniso": {
        "muon": {  # For Mu50 (| TkMu50 )
            "2016preVFP": "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2016postVFP": "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2017": "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2018": "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
        },
    },
    "isolation": {
        "muon": {
            "2016preVFP": "NUM_LooseRelIso_DEN_MediumPromptID",
            "2016postVFP": "NUM_LooseRelIso_DEN_MediumPromptID",
            "2017": "NUM_LooseRelIso_DEN_MediumPromptID",
            "2018": "NUM_LooseRelIso_DEN_MediumPromptID",
        },
    },
    "id": {
        "muon": {
            "2016preVFP": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2016postVFP": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2017": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2018": "NUM_MediumPromptID_DEN_TrackerMuons",
        },
        "electron": {
            "2016preVFP": "wp90noiso",
            "2016postVFP": "wp90noiso",
            "2017": "wp90noiso",
            "2018": "wp90noiso",
        },
    },
    "reco": {
        "electron": {
            "2016preVFP": "RecoAbove20",
            "2016postVFP": "RecoAbove20",
            "2017": "RecoAbove20",
            "2018": "RecoAbove20",
        },
    },
}

def add_lepton_weight(weights, lepton, year, lepton_type="muon"):
    ul_year = f"{year}_UL"
    if lepton_type == "electron":
        ul_year = ul_year.replace("_UL", "")

    cset = correctionlib.CorrectionSet.from_file(get_pog_json(lepton_type, year))

    def set_isothreshold(corr, value, lepton_pt, lepton_type):
        """
        restrict values to 1 for some SFs if we are above/below the ISO threshold
        """
        iso_threshold = {"muon": 55.0, "electron": 120.0}[lepton_type]
        if corr == "trigger_iso":
            value[lepton_pt > iso_threshold] = 1.0
        elif corr == "trigger_noniso":
            value[lepton_pt < iso_threshold] = 1.0
        elif corr == "isolation":
            value[lepton_pt > iso_threshold] = 1.0
        elif corr == "id" and lepton_type == "electron":
            value[lepton_pt < iso_threshold] = 1.0
        return value

    def get_clip(lep_pt, lep_eta, lepton_type, corr=None):
        clip_pt = [0.0, 2000]
        clip_eta = [-2.4999, 2.4999]
        if lepton_type == "electron":
            clip_pt = [10.0, 499.999]
            if corr == "reco":
                clip_pt = [20.1, 499.999]
        elif lepton_type == "muon":
            clip_pt = [30.0, 1000.0]
            clip_eta = [0.0, 2.3999]
            if corr == "trigger_noniso":
                clip_pt = [52.0, 1000.0]
        lepton_pt = np.clip(lep_pt, clip_pt[0], clip_pt[1])
        lepton_eta = np.clip(lep_eta, clip_eta[0], clip_eta[1])
        return lepton_pt, lepton_eta

    lep_pt = np.array(ak.fill_none(lepton.pt, 0.0))
    lep_eta = np.array(ak.fill_none(lepton.eta, 0.0))
    if lepton_type == "muon":
        lep_eta = np.abs(lep_eta)

    for corr, corrDict in lepton_corrections.items():
        if lepton_type not in corrDict.keys():
            continue
        if year not in corrDict[lepton_type].keys():
            continue

        json_map_name = corrDict[lepton_type][year]

        lepton_pt, lepton_eta = get_clip(lep_pt, lep_eta, lepton_type, corr)

        values = {}
        if lepton_type == "muon":
            values["nominal"] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, "nominal")
        else:
            values["nominal"] = cset["UL-Electron-ID-SF"].evaluate(ul_year, "sf", json_map_name, lepton_eta, lepton_pt)

        if (lepton_type == "muon") and (corr == "id"):
            # split the stat. and syst. unc. for the id SF for muons
            for unc_type in ["stat", "syst"]:
                values[unc_type] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, unc_type)
                for key, val in values.items():
                    values[key] = set_isothreshold(corr, val, np.array(ak.fill_none(lepton.pt, 0.0)), lepton_type)

                up = values["nominal"] * (1 + values[unc_type])
                down = values["nominal"] * (1 - values[unc_type])
                weights.add(f"{corr}_{lepton_type}_{unc_type}", values["nominal"], up, down)

        else:
            if lepton_type == "muon":
                values["up"] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, "systup")
                values["down"] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, "systdown")
            else:
                values["up"] = cset["UL-Electron-ID-SF"].evaluate(ul_year, "sfup", json_map_name, lepton_eta, lepton_pt)
                values["down"] = cset["UL-Electron-ID-SF"].evaluate(ul_year, "sfdown", json_map_name, lepton_eta, lepton_pt)

            for key, val in values.items():
                values[key] = set_isothreshold(corr, val, np.array(ak.fill_none(lepton.pt, 0.0)), lepton_type)

            # add weights (for now only the nominal weight)
            weights.add(f"{corr}_{lepton_type}", values["nominal"], values["up"], values["down"])


def get_pileup_weight(year: str, nPU: np.ndarray):
    """
    Should be able to do something similar to lepton weight but w pileup
    e.g. see here: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/LUMI_puWeights_Run2_UL/
    """
    cset = correctionlib.CorrectionSet.from_file('utils/puWeights.json.gz')

    year_to_corr = {
        "2016preVFP": "Collisions16_UltraLegacy_goldenJSON",
        "2016postVFP": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
    }

    values = {}
    values["nominal"] = cset[year_to_corr[year]].evaluate(nPU, "nominal")
    values["up"] = cset[year_to_corr[year]].evaluate(nPU, "up")
    values["down"] = cset[year_to_corr[year]].evaluate(nPU, "down")

    return values


def add_pileup_weight(weights: Weights, year: str, nPU: np.ndarray):
    """Separate wrapper function in case we just want the values separately."""
    values = get_pileup_weight(year, nPU)
    weights.add("pileup", values["nominal"], values["up"], values["down"])


def add_pileupid_weights(weights: Weights, jets: JetArray, genjets, wp: str = "L"):
    """Pileup ID scale factors
    https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL#Data_MC_Efficiency_Scale_Factors

    Takes ak4 jets which already passed the pileup ID WP.
    Only applies to jets with pT < 50 GeV and those geometrically matched to a gen jet.
    """

    # pileup ID should only be used for jets with pT < 50
    jets = jets[(jets.pt < 50) & (jets.pt > 12.5)]
    # check that there's a geometrically matched genjet (99.9% are, so not really necessary...)
    jets = jets[ak.any(jets.metric_table(genjets) < 0.4, axis=-1)]

    sf_cset = correctionlib.CorrectionSet.from_file('utils/jmar.json.gz')["PUJetID_eff"]

    # save offsets to reconstruct jagged shape
    offsets = jets.pt.layout.offsets

    sfs_var = []
    for var in ["nom", "up", "down"]:
        # correctionlib < 2.3 doesn't accept jagged arrays (but >= 2.3 needs awkard v2)
        sfs = sf_cset.evaluate(ak.flatten(jets.eta), ak.flatten(jets.pt), var, wp)
        # reshape flat effs
        sfs = ak.Array(ak.layout.ListOffsetArray64(offsets, ak.layout.NumpyArray(sfs)))
        # product of SFs across arrays, automatically defaults empty lists to 1
        sfs_var.append(ak.prod(sfs, axis=1))

    weights.add("pileupIDSF", *sfs_var)