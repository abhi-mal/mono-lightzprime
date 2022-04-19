version = '2018'

# lumi here are in pb^-1 

lumi = {"SingleEleCR":59699,
        "DoubleEleCR":59699,
        "SingleMuCR":59699,
        "DoubleMuCR":59699,
        "GammaCR":59699,
        #"SignalRegion":59699/5.
        "SignalRegion":59699
}

lumi_by_era = {"SingleEleCR":{"A":14024,"B":7061,"C":6895,"D":31720},
               "DoubleEleCR":{"A":14024,"B":7061,"C":6895,"D":31720},
               "SingleMuCR":{"A":14024,"B":7061,"C":6895,"D":31720},
               "DoubleMuCR":{"A":14024,"B":7061,"C":6895,"D":31720},
               "GammaCR":{"A":14024,"B":7061,"C":6895,"D":31720},
               #"SignalRegion":{"A":14024/5.,"B":7061/5.,"C":6895/5.,"D":31720/5.},
               "SignalRegion":{"A":14024,"B":7061,"C":6895,"D":31720}
}
# xsecs with key == 'gen' are from GenXSecAnalyzer, bkg_xsec taken from monojet AN, signal_xsec taken from ealier monozprime
xsec_bkgs = {
        ######2018#####
        ######DY1Jets##### (NLO in QCD)
        'DY1JetsToLL_M-50_LHEZpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 315.1} ,
        'DY1JetsToLL_M-50_LHEZpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 9.5} ,
        'DY1JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 1.097} ,
        'DY1JetsToLL_M-50_LHEZpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 0.1207} ,
        ######DY2Jets##### (NLO in QCD)
        'DY2JetsToLL_M-50_LHEZpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 169} ,
        'DY2JetsToLL_M-50_LHEZpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 15.73} ,
        'DY2JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 2.74} ,
        'DY2JetsToLL_M-50_LHEZpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 0.4492} ,
        ######GJets##### (NLO in QCD)
        'GJets_1j_Gpt-50To100_5f_NLO_Autumn18' :  {'gen': 14165.0}, # FIXME files missing!,
        'GJets_1j_Gpt-100To250_5f_NLO_Autumn18' :  {'gen': 1183.0},
        #'GJets_1j_Gpt-100To250_5f_NLO_Autumn18_5' :  {'gen': 1183.0},
        'GJets_1j_Gpt-250To400_5f_NLO_Autumn18' :  {'gen': 26.09} ,
        'GJets_1j_Gpt-400To650_5f_NLO_Autumn18' :  {'gen': 3.148} ,
        'GJets_1j_Gpt-650ToInf_5f_NLO_Autumn18' :  {'gen': 0.2887} ,        
        ######WJets##### (NLO in QCD)
        'WJetsToLNu_Pt-50To100_TuneCP5_13TeV-amcatnloFXFX-pythia8' :  {'gen': 3569.0} ,
        'WJetsToLNu_Pt-100To250_TuneCP5_13TeV-amcatnloFXFX-pythia8' :  {'gen': 769.8},
        #'WJetsToLNu_Pt-100To250_TuneCP5_13TeV-amcatnloFXFX-pythia8_ext1' :  {'gen': 769.8},
        #'WJetsToLNu_Pt-250To400_TuneCP5_13TeV-amcatnloFXFX-pythia8_ext1' :  {'gen': 27.86} ,
        'WJetsToLNu_Pt-250To400_TuneCP5_13TeV-amcatnloFXFX-pythia8' :  {'gen': 27.62} ,
        #'WJetsToLNu_Pt-400To600_TuneCP5_13TeV-amcatnloFXFX-pythia8_ext1' :  {'gen': 3.591} ,
        'WJetsToLNu_Pt-400To600_TuneCP5_13TeV-amcatnloFXFX-pythia8' :  {'gen': 3.591} ,        
        #'WJetsToLNu_Pt-600ToInf_TuneCP5_13TeV-amcatnloFXFX-pythia8_ext1' :  {'gen': 0.549} ,
        'WJetsToLNu_Pt-600ToInf_TuneCP5_13TeV-amcatnloFXFX-pythia8' :  {'gen': 0.549} ,        
        ######QCD##### (LO in QCD)
        'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8' :  {'gen': 1.85e+08} ,
        'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8' :  {'gen': 2.369e+07} ,
        'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8' :  {'gen': 1.554e+06} ,
        'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8' :  {'gen': 324300.0} ,
        'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8' :  {'gen': 29990.0} ,
        'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8' :  {'gen': 6374.0} ,
        'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8' :  {'gen': 1095.0} ,
        'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8' :  {'gen': 99.27} ,
        'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8' :  {'gen': 20.25} ,
        ######Z1Jets##### (NLO in QCD)
        'Z1JetsToNuNu_M-50_LHEZpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 598.9} ,
        'Z1JetsToNuNu_M-50_LHEZpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 18.04} ,        
        'Z1JetsToNuNu_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 2.051} ,
        'Z1JetsToNuNu_M-50_LHEZpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 0.2251} ,        
        ######Z2Jets##### (NLO in QCD)
        'Z2JetsToNuNu_M-50_LHEZpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 326.3} ,
        'Z2JetsToNuNu_M-50_LHEZpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 29.6},
        'Z2JetsToNuNu_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 5.174} ,
        'Z2JetsToNuNU_M-50_LHEZpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8' :  {'gen': 0.8472}, 
        #######TTJets#### (NLO in QCD)
        'TTJets' : {'gen': 831.76},
        #######ST_t-channel_top_4f###### (NLO in QCD)
        'ST_t-channel_top_4f_InclusiveDecays_TuneCP5' : {'gen': 137.458},
        #######ST_t-channel_antitop_4f###### (NLO in QCD)
        'ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5' : {'gen': 83.0066},
        #######ST_tW_top_5f###### (NLO in QCD)
        'ST_tW_top_5f_inclusiveDecays_TuneCP5' : {'gen': 35.85},
        #######ST_tW_antitop_5f###### (NLO in QCD)
        'ST_tW_antitop_5f_inclusiveDecays_TuneCP5' : {'gen': 35.85},
        ####### WW ###### (LO in QCD)
        'WW' : {'gen': 75.91},
        ####### WZ ###### (LO in QCD)
        'WZ' : {'gen': 27.56},
        ####### ZZ ###### (LO in QCD)
        'ZZ' : {'gen': 12.14},
}

xsec_sigs = {
        ###### Mono-LightZ'#####
       'Mx1Mv1000' : {
               'gen': 9.520e-06,
               'theo': 1.499600E+00, 
                },
}

rebin_values = {
        #Associatedfatjet_vars
        'Associatedfatjet_area' : 1,
        'Associatedfatjet_eta' : 2,
        'Associatedfatjet_phi' : 2,
        'Associatedfatjet_pt' : 50,
        'Associatedfatjet_pt_raw': 50,
        'Associatedfatjet_pt_nom': 50,    
        'Associatedfatjet_mass' : 10,
        'Associatedfatjet_mass_raw': 10,
        'Associatedfatjet_mass_nom': 10,
        'Associatedfatjet_msoftdrop' : 10,
        'Associatedfatjet_msoftdrop_raw':10,
        'Associatedfatjet_msoftdrop_nom':10,
        'Associatedfatjet_msoftdrop_corr_PUPPI':10,
        'Associatedfatjet_msoftdrop_corr_JMS':10,
        'Associatedfatjet_msoftdrop_corr_JMR':10,
        'Associatedfatjet_msoftdrop_tau21DDT_nom':10,
        'Associatedfatjet_lsf3' : 1,
        'Associatedfatjet_hadronflavour':1,
        'Associatedfatjet_nBHadrons':1,
        'Associatedfatjet_nCHadrons':1,
        'Associatedfatjet_n2b1' : 1,
        'Associatedfatjet_n3b1' : 1,
        'Associatedfatjet_tau1' : 1,
        'Associatedfatjet_tau2' : 1,
        'Associatedfatjet_tau3' : 1,
        'Associatedfatjet_tau4' : 1,
        'Associatedfatjet_tau2bytau1' : 1,
        'Associatedfatjet_tau3bytau2' : 2,
        'Associatedfatjet_deepTag_ZvsQCD':1,
        'Associatedfatjet_deepTagMD_ZvsQCD':1,
        'Associatedfatjet_particleNetMD_Xqq':1,
        'Associatedfatjet_particleNetMD_QCD':1,
        'Associatedfatjet_particleNet_ZvsQCD':1,
        'Associatedfatjet_deepTag_ZvsQCD':1,
        'Associatedfatjet_deepTagMD_ZvsQCD':1,
        'Associatedfatjet_corr_JEC':1,
        'Associatedfatjet_corr_JER':1,
        'Associatedfatjet_corr_JMS':1,
        'Associatedfatjet_corr_JMR':1,
        #associatedjet_vars
        'Associatedjet_pt' : 50,
        'Associatedjet_pt_raw' : 50,
        'Associatedjet_pt_nom' : 50,
        'Associatedjet_eta' : 2,
        'Associatedjet_phi' : 2,
        'Associatedjet_area' : 1,
        'Associatedjet_jetId' : 1,
        'Associatedjet_chHEF' : 1,
        'Associatedjet_neHEF' : 1,
        'Associatedjet_neEmEF' : 1,
        'Associatedjet_mass' : 10,
        'Associatedjet_mass_raw' : 10,
        'Associatedjet_mass_nom' : 10,
        'Associatedjet_nConstituents' : 1,
        'Associatedjet_corr_JEC':1,
        'Associatedjet_corr_JER':1,    
        #HPSJet_vars
        'HPSJet_charge' : 1,
        'HPSJet_pt' : 50,
        'HPSJet_mass' : 1,
        'HPSJet_eta' : 2,
        'HPSJet_phi' : 2,        
        'HPSJet_leadTkDeltaEta' : 2,
        'HPSJet_leadTkDeltaPhi' : 2,
        'HPSJet_photonsOutsideSignalCone' : 1, 
        'HPSJet_leadTkPtOverhpsPt':5,
        'HPSJet_decaymode':1,
        #my_vars
        'HPSJet_AssociatedJet_DeltaEta':2,
        'HPSJet_AssociatedJet_DeltaPhi':2,
        'HPSJet_AssociatedFatJet_DeltaEta':2,
        'HPSJet_AssociatedFatJet_DeltaPhi':2,        
        'leadTkPtOverAssociatedJetPt' : 2,
        'HPSJetPtOverAssociatedJetPt' : 5,
        'HPSJetPtOverAssociatedFatJetPt' : 5,
        'HPSJetChargeForDecayModes_5and6' : 1,               
        #general
        'cutflow' : 1,
        'cutflow_scaled': 1,
        'argmaxHPSJet' : 1,
        'recoil' : 50,
        'muon1_pt':50,
        'muon2_pt':50,
        'muon1_eta':2,
        'muon2_eta':2,
        'muon1_phi':2,
        'muon2_phi':2,
        'dimuon_pt':50,
        'dimuon_eta':2,
        'dimuon_phi':2,
}