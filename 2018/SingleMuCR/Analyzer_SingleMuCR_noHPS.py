#!/usr/bin/env python
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.modules.common.hpsjetCorrProducer import HPSJetCorrs2018ReReco
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.jetmetHelperRun2 import *
from importlib import import_module
import os
import sys
import ROOT
import math,itertools
import argparse
ROOT.PyConfig.IgnoreCommandLineOptions = True

file_out = open("my_output_log.txt","w")

class SingleMuCR_noHPS(Module):
    def __init__(self,is_data,process,year):
        #self.writeHistFile = True
        self.is_data = is_data
        self.processname = process
        self.year = year
        self.sfmap = {}; self.sf_files = {}; # we need to keep the files around because otherwise root deletes it internally and we get an error
        self.event_weights= dict.fromkeys(["met_trig","PU","gen_wt","nlo_ewk","nlo_qcd_ewk","nnlo_qcd","id_iso","hpsjet_sf","all"],1)
        self.getSFhists()
        pass

    def getSFhists(self): 
           #file_out.write("Going to get the SF hiistograms\n")
           if self.processname in ["Z1Jets","Z2Jets","WJets","GJets","DY1Jets","DY2Jets"]:  
                   # get kfactor hists
                   #in_path = "/nfs_scratch/mallampalli/Analysis/Mono-zprime/nanoAOD_test/analysis_trial/CMSSW_10_6_30/src/analysis/"
                   #in_path = ""
                   self.sf_files["f_nnlo_qcd"] = ROOT.TFile.Open('RootFiles/theory/lindert_qcd_nnlo_sf.root','r')
                   #file_out.write("self.sf_files['f_nnlo_qcd']=%s, type=%s\n"%(self.sf_files["f_nnlo_qcd"],type(self.sf_files["f_nnlo_qcd"])))
                   if (self.processname == "WJets"): 
                           self.sf_files["f_nlo_ewk"] = ROOT.TFile.Open('RootFiles/theory/merged_kfactors_wjets.root','r')
                           self.sfmap["nnlo_qcd"] = self.sf_files["f_nnlo_qcd"].Get("evj")
                           self.sfmap["nlo_ewk"] = self.sf_files["f_nlo_ewk"].Get("kfactor_monojet_ewk")
                           self.sfmap["nlo_qcd_ewk"] = self.sf_files["f_nlo_ewk"].Get("kfactor_monojet_qcd_ewk")
                   if (self.processname in ["Z1Jets","Z2Jets","DY1Jets","DY2Jets"]): 
                           self.sf_files["f_nlo_ewk"] = ROOT.TFile.Open('RootFiles/theory/merged_kfactors_zjets.root','r')
                           self.sfmap["nnlo_qcd"] = self.sf_files["f_nnlo_qcd"].Get("eej")
                           self.sfmap["nlo_ewk"] = self.sf_files["f_nlo_ewk"].Get("kfactor_monojet_ewk")
                           self.sfmap["nlo_qcd_ewk"] = self.sf_files["f_nlo_ewk"].Get("kfactor_monojet_qcd_ewk")
                   if (self.processname == "GJets"): 
                           self.sf_files["f_nlo_ewk"] = ROOT.TFile.Open('RootFiles/theory/merged_kfactors_gjets.root','r')
                           self.sfmap["nnlo_qcd"] = self.sf_files["f_nnlo_qcd"].Get("aj")
                           self.sfmap["nlo_ewk"] = self.sf_files["f_nlo_ewk"].Get("kfactor_monojet_ewk")
                           #self.sfmap["nlo_qcd_ewk"] = f_nlo_ewk.Get("")                           
           # get trigger_sf hists
           self.sf_files["f_met_trigger_sf"] =  ROOT.TFile.Open('RootFiles/trigger/met_trigger_sf.root','r')
           self.sfmap["met_trig"] = self.sf_files["f_met_trigger_sf"].Get("120pfht_hltmu_1m_2018")
           # get pu_weights hists
           self.sf_files["f_pu"] = ROOT.TFile.Open('RootFiles/pileup/PU_Central_2018.root','r')
           self.sfmap["PU"] = self.sf_files["f_pu"].Get("pileup")
           # get Muon Id and Iso SF hists
           self.sf_files["f_muon_id"] = ROOT.TFile.Open('RootFiles/muon/2018_RunABCD_SF_ID.root','r')
           self.sf_files["f_muon_iso"]= ROOT.TFile.Open('RootFiles/muon/2018_RunABCD_SF_ISO.root','r')
           self.sfmap["muon_id_loose"]= self.sf_files["f_muon_id"].Get("NUM_LooseID_DEN_TrackerMuons_pt_abseta")
           self.sfmap["muon_id_tight"]= self.sf_files["f_muon_id"].Get("NUM_TightID_DEN_TrackerMuons_pt_abseta")
           self.sfmap["muon_iso_loose"]= self.sf_files["f_muon_iso"].Get("NUM_LooseRelIso_DEN_LooseID_pt_abseta")
           self.sfmap["muon_iso_tight"]= self.sf_files["f_muon_iso"].Get("NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta")
           #file_out.write("\n%s\n"%self.sfmap)
           #file_out.write("Loaded all the SF hiistograms\n")

    def get_dilep_cands(self,lep_collection,tag):
            dilep_cands = []
            lep_pairs = list(itertools.combinations(lep_collection,2))
            for lep_pair in lep_pairs:
                    if(tag == "Z"): cond = (lep_pair[0].pdgId + lep_pair[1].pdgId ==0) # same flavor opp sign
                    if(tag == "W"): cond = (lep_pair[0].pdgId != lep_pair[1].pdgId) # the colllection anyways has same flavour only
                    if(cond): 
                            lep_comb = lep_pair[0].p4() + lep_pair[1].p4()
                            dilep_cands.append(lep_comb)   
            return dilep_cands                         

    def GenNeutrinos(self,neutrinos,zmass):
            #file_out.write("Didn't find a gen Z, so getting from neutrinos\n")
            bosonPt = 0; diff = 9999
            dilep_cands = self.get_dilep_cands(neutrinos,"Z")
            if(len(dilep_cands)==0):  bosonPt = -1 
            else:         
                    for x in dilep_cands: 
                            if(abs(x.M()-zmass)<diff):   best_dilepton = x;   diff = abs(x.M()-zmass)  
                    bosonPt = best_dilepton.Pt()
            return bosonPt

    def GenDilepton(self,gen_particles,dressed_leps,zmass):
            #file_out.write("Didn't find a gen Z, so getting from charged leptons\n")
            bosonPt = 0; diff = 9999
            # from leps, since DoubleMuCR
            dressed_muons = [y for y in dressed_leps if abs(y.pdgId)==13] # remove muon neutrinos
            dilep_cands_mu = self.get_dilep_cands(dressed_muons,"Z")          
            # from taus
            gen_taus = [y for y in gen_particles if (abs(y.pdgId)==15 and y.status==2)] # taus have status 2
            dilep_cands_tau = self.get_dilep_cands(gen_taus,"Z")
            dilep_cands = dilep_cands_mu + dilep_cands_tau
            if(len(dilep_cands)==0):  bosonPt = -1
            else:
                    for x in dilep_cands: 
                            if(abs(x.M()-zmass)<diff):   best_dilepton = x;   diff = abs(x.M()-zmass)
                    bosonPt = best_dilepton.Pt()
            return bosonPt

    def GenDileptonW(self,gen_particles,dressed_leps,wmass):
            #file_out.write("Didn't find a gen W, so getting from leptons\n")
            bosonPt = 0; diff = 9999
            mu_nu_collection = [y for y in dressed_leps]
            dilep_cands_mu = self.get_dilep_cands(mu_nu_collection,"W") 
            tau_nu_collection =  [y for y in gen_particles if ((abs(y.pdgId)==15 and y.status==2) or (abs(y.pdgId)==16))]
            dilep_cands_tau = self.get_dilep_cands(tau_nu_collection,"W")
            dilep_cands = dilep_cands_mu + dilep_cands_tau
            if(len(dilep_cands)==0):  bosonPt = -1 
            else:
                    for x in dilep_cands: 
                            if(abs(x.M()-wmass)<diff):   best_dilepton = x;   diff = abs(x.M()-wmass)     
                    bosonPt = best_dilepton.Pt()
            return bosonPt

    def get_genboson_pt(self, event): 
            #file_out.write(" Getting gen Boson Pt\n")
            if (self.processname == "WJets"): boson_pdg_id = 24 #W
            if (self.processname in ["Z1Jets","Z2Jets","DY1Jets","DY2Jets"]): boson_pdg_id = 23 #Z
            if (self.processname == "GJets"): boson_pdg_id = 22 #photon
            bosonPt = 0; good_bosons = []; neutrinos = []; zmass = 91; wmass = 81
            gen_particles = Collection(event, "GenPart")
            for genPart in gen_particles:# nGenPart= interesting gen partiles
                    if((abs(genPart.pdgId) in [12,14,16]) and(genPart.status == 1)): neutrinos.append(genPart) 
                    if(abs(genPart.pdgId) != boson_pdg_id): continue
                    if (self.processname == "GJets"): 
                            if(genPart.status == 1): 
                                    if(genPart.statusFlags&1==1): good_bosons.append(genPart)
                                    #else: file_out.write("GJets: status 1 but status flag bit 0 is not 1!!")
                    else:
                            if(genPart.status== 62): good_bosons.append(genPart)
                    #file_out.write("processname: %s | status =%d"%(self.processname,genPart.status))   
            if(len(good_bosons)!=0):
                    best_boson =  sorted(good_bosons, key=lambda x : x.pt, reverse=True)[0] 
                    bosonPt =  best_boson.pt
            else:   
                    dressed_leps = Collection(event,"GenDressedLepton"); dressed_mus =[]
                    for x in dressed_leps: # DoubleMuCR: so leptons should be muons, muon neutrinos
                            if(abs(x.pdgId)==13 or abs(x.pdgId)==14): dressed_mus.append(x)
                    if(self.processname in ["Z1Jets","Z2Jets"]): bosonPt = self.GenNeutrinos(neutrinos,zmass)
                    elif(self.processname in ["DY1Jets","DY2Jets"]): bosonPt = self.GenDilepton(gen_particles,dressed_mus,zmass)
                    elif(self.processname == "WJets"):   bosonPt = self.GenDileptonW(gen_particles,dressed_mus,wmass)
            #file_out.write(" Got gen Boson Pt\n")        
            return  bosonPt                            

    def get_sf_from_hist(self,histo,value,tag):
            sf =1 
            xmax = histo.GetXaxis().GetXmax(); xmin = histo.GetXaxis().GetXmin()
            if ( value >= xmax ): pass #xbin = histo.GetNbinsX() not applying sf if outside range
            elif( value <= xmin ): pass #xbin = 1 not applying sf if outside range, because puhist bin 0 is junk!
            else: xbin = histo.GetXaxis().FindBin(value); sf = histo.GetBinContent( xbin )
            #file_out.write("Process: %s | reweight: %s | sf: %s\n"%(self.processname,tag,sf))
            return sf

    def get_sf_from_hist_2D(self,histo,value_x,value_y,tag):
            sf =1 ; xbin = -99 ; ybin = -99
            xmax = histo.GetXaxis().GetXmax(); xmin = histo.GetXaxis().GetXmin()
            if ( value_x >= xmax ): pass #xbin = histo.GetNbinsX() not applying sf if outside range
            elif( value_x <= xmin ): pass #xbin = 1 not applying sf if outside range, because puhist bin 0 is junk!
            else: xbin = histo.GetXaxis().FindBin(value_x)
            ymax = histo.GetYaxis().GetXmax(); ymin = histo.GetYaxis().GetXmin() # get the Y axis and use the funcs GetXmax and GetXmin
            if ( value_y >= ymax ): pass #xbin = histo.GetNbinsX() not applying sf if outside range
            elif( value_y <= ymin ): pass #xbin = 1 not applying sf if outside range, because puhist bin 0 is junk!
            else: ybin = histo.GetYaxis().FindBin(value_y)
            if (xbin != -99 and ybin != -99): sf = histo.GetBinContent( xbin,ybin )
            #file_out.write("Process: %s | reweight: %s | sf: %s\n"%(self.processname,tag,sf))
            return sf

    def apply_id_iso_weights(self, event, event_weight,selected_muon):
            mu_tightid_sf = self.get_sf_from_hist_2D(self.sfmap["muon_id_tight"],selected_muon.pt,abs(selected_muon.eta),"selected_muon_tight_id")
            mu_tightiso_sf= self.get_sf_from_hist_2D(self.sfmap["muon_iso_tight"],selected_muon.pt,abs(selected_muon.eta),"selected_muon_tight_iso")      
            self.event_weights["id_iso"] = mu_tightid_sf*mu_tightiso_sf
            event_weight = self.event_weights["id_iso"] * event_weight
            return event_weight


    def apply_pu_weights(self, event, event_weight): 
            #file_out.write(" Starting PU event weight calculation\n")
            self.event_weights["PU"] = self.get_sf_from_hist(self.sfmap["PU"],event.Pileup_nTrueInt,"PU")
            event_weight = self.event_weights["PU"] * event_weight
            #file_out.write(" Finished PU event weight calculation\n")
            return event_weight

    def apply_gen_weights(self,event,event_weight):
            if(abs(event.Generator_weight)>0): self.event_weights["gen_wt"] = event.Generator_weight/abs(event.Generator_weight)# some events can have -ve weights!
            else: self.event_weights["gen_wt"] = 0
            event_weight = self.event_weights["gen_wt"] * event_weight
            return event_weight                    

    def apply_trigger_sf(self, event, event_weight): 
            #file_out.write(" Starting Trigger event weight calculation\n")
            self.event_weights["met_trig"] = self.get_sf_from_hist(self.sfmap["met_trig"],event.recoil,"MET_trigger")
            event_weight = self.event_weights["met_trig"] * event_weight
            #file_out.write(" Finished Trigger event weight calculation\n")
            return event_weight

    def apply_kfactors(self, event, event_weight): 
            #file_out.write(" Starting kFactor event weight calculation\n")
            boson_pt = self.get_genboson_pt(event)
            if (boson_pt==-1): return event_weight
            self.event_weights["nnlo_qcd"] = self.get_sf_from_hist(self.sfmap["nnlo_qcd"],boson_pt,"nnlo_qcd")
            #file_out.write(" Finished nnlo qcd event weight calculation\n")
            self.event_weights["nlo_ewk"] = self.get_sf_from_hist(self.sfmap["nlo_ewk"],boson_pt,"nlo_ewk")
            #file_out.write(" Finished nlo ewk event weight calculation\n")
            if self.processname in ["GJets"]: self.event_weights["nlo_qcd_ewk"] = 1
            else: self.event_weights["nlo_qcd_ewk"] = self.get_sf_from_hist(self.sfmap["nlo_qcd_ewk"],boson_pt,"nlo_qcd_ewk")
            #file_out.write(" Finished nlo qcd ewk event weight calculation\n")
            event_weight = self.event_weights["nnlo_qcd"] *  self.event_weights["nlo_ewk"] * self.event_weights["nlo_qcd_ewk"] * event_weight
            #file_out.write(" Finished kFactor event weight calculation\n")
            return event_weight

    def apply_hpsjet_sf(self,hpsjet,event_weight):
            self.event_weights["hpsjet_sf"] = hpsjet.sfDeepTau2017v2p1VSjet_VVVLoose
            event_weight = self.event_weights["hpsjet_sf"] * event_weight
            return event_weight

    def get_event_weight(self, event,selected_muons): 
            #file_out.write("Starting event weight calculation\n")
            event_weight = 1
            #self.getSFhists()
            #file_out.write("Got event weight hists\n")
            event_weight = self.apply_id_iso_weights(event,event_weight,selected_muons)
            event_weight = self.apply_pu_weights(event,event_weight)
            event_weight = self.apply_trigger_sf(event,event_weight)
            if self.processname in ["Z1Jets","Z2Jets","WJets","GJets","DY1Jets","DY2Jets"]: event_weight = self.apply_kfactors(event,event_weight)
            return event_weight

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        self.debug_vars = {}
        #AssociatedJet(selected from Ak4Jets)
        self.out.branch("AK4Jet_pt"  , "F"); self.out.branch("AK4Jet_mass"  , "F");  self.out.branch("AK4Jet_nConstituents"  , "F")
        self.out.branch("AK4Jet_eta" , "F"); self.out.branch("AK4Jet_chHEF"  , "F"); self.out.branch("AK4Jet_neEmEF"  , "F")
        self.out.branch("AK4Jet_phi" , "F"); self.out.branch("AK4Jet_jetId"  , "F"); self.out.branch("AK4Jet_neHEF"  , "F")
        self.out.branch("AK4Jet_area", "F")
        #AssociatedFatJet
        self.out.branch("Associatedfatjet_index", "I")
        # General
        self.out.branch("recoil" , "F") 
        self.out.branch("muon_pt" , "F"); self.out.branch("muon_eta" , "F"); self.out.branch("muon_phi" , "F")
        #event weights
        self.out.branch("evt_wt_met_trig" , "F"); self.out.branch("evt_wt_PU" , "F"); self.out.branch("evt_wt_gen_wt" , "F"); self.out.branch("evt_wt_nlo_ewk" , "F"); self.out.branch("evt_wt_nlo_qcd_ewk" , "F"); self.out.branch("evt_wt_nnlo_qcd" , "F"); self.out.branch("evt_wt_mu_id_iso" , "F"); self.out.branch("evt_wt_hpsjet_sf" , "F"); self.out.branch("evt_wt_all" , "F")        

        self.cutflow = ROOT.TH1F('cutflow', 'cutflow', 17, 0, 17)
        #event related
        self.cutflow.GetXaxis().SetBinLabel(1,"total_events")
        self.cutflow.GetXaxis().SetBinLabel(2,"passing_trigger")
        self.cutflow.GetXaxis().SetBinLabel(3,"MET_filters")
        self.cutflow.GetXaxis().SetBinLabel(4,"dphi_PFMET_TkMET")
        self.cutflow.GetXaxis().SetBinLabel(5,"dMET_PF_Calo")         
        self.cutflow.GetXaxis().SetBinLabel(6,"muon_CRselection")
        self.cutflow.GetXaxis().SetBinLabel(7,"lepMET_MtCut")
        self.cutflow.GetXaxis().SetBinLabel(8,"electron_veto")         
        self.cutflow.GetXaxis().SetBinLabel(9,"photon_veto") 
        #self.cutflow.GetXaxis().SetBinLabel(10,"bjet_veto")
        self.cutflow.GetXaxis().SetBinLabel(11-1,"atleast_1_ak4jet")
        #HPSJet related
        self.cutflow.GetXaxis().SetBinLabel(12-1,"atleast_1_cleanak4jet")         
        self.cutflow.GetXaxis().SetBinLabel(13-1,"dphi_AK4Jet_MET") 
        self.cutflow.GetXaxis().SetBinLabel(14-1,"AK4Jet_ptetacuts")
        #self.cutflow.GetXaxis().SetBinLabel(14-1,"wjets_cleaning")
        #self.cutflow.GetXaxis().SetBinLabel(15-1,"zjets_cleaning") 
        self.cutflow.GetXaxis().SetBinLabel(17-3,"AK4Jet_tightLepVeto")
        self.cutflow.GetXaxis().SetBinLabel(18-3,"has_associatedfatJet") 
        self.cutflow.GetXaxis().SetBinLabel(19-3,"recoil_cut")
        self.cutflow.GetXaxis().SetBinLabel(20-3,"associatedfatJet_has2subjets")
        self.cutflow_events = dict.fromkeys({"total_events","passing_trigger","MET_filters","dphi_PFMET_TkMET","dMET_PF_Calo","muon_CRselection","lepMET_MtCut","electron_veto","photon_veto","atleast_1_ak4jet","atleast_1_cleanak4jet","dphi_AK4Jet_MET","AK4Jet_ptetacuts","AK4Jet_tightLepVeto","has_associatedfatJet","recoil_cut","associatedfatJet_has2subjets"},0)
        self.argmaxAK4Jet = ROOT.TH1F('argmaxAK4Jet', 'argmaxAK4Jet', 11, 0, 11)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        #event related
        self.cutflow.SetBinContent(1,self.cutflow_events["total_events"])
        self.cutflow.SetBinContent(2,self.cutflow_events["passing_trigger"])
        self.cutflow.SetBinContent(3,self.cutflow_events["MET_filters"])
        self.cutflow.SetBinContent(4,self.cutflow_events["dphi_PFMET_TkMET"])
        self.cutflow.SetBinContent(5,self.cutflow_events["dMET_PF_Calo"])        
        self.cutflow.SetBinContent(6,self.cutflow_events["muon_CRselection"])
        self.cutflow.SetBinContent(7,self.cutflow_events["lepMET_MtCut"])
        self.cutflow.SetBinContent(8,self.cutflow_events["electron_veto"])
        self.cutflow.SetBinContent(9,self.cutflow_events["photon_veto"]) 
        #self.cutflow.SetBinContent(10,self.cutflow_events["bjet_veto"]) 
        self.cutflow.SetBinContent(11-1,self.cutflow_events["atleast_1_ak4jet"]) 
        self.cutflow.SetBinContent(12-1,self.cutflow_events["atleast_1_cleanak4jet"])
        self.cutflow.SetBinContent(13-1,self.cutflow_events["dphi_AK4Jet_MET"])
        #HPSJet related
        self.cutflow.SetBinContent(14-1,self.cutflow_events["AK4Jet_ptetacuts"])
        #self.cutflow.SetBinContent(15-1,self.cutflow_events["wjets_cleaning"])
        #self.cutflow.SetBinContent(16-1,self.cutflow_events["zjets_cleaning"])
        self.cutflow.SetBinContent(17-3,self.cutflow_events["AK4Jet_tightLepVeto"]) 
        self.cutflow.SetBinContent(18-3,self.cutflow_events["has_associatedfatJet"])
        self.cutflow.SetBinContent(19-3,self.cutflow_events["recoil_cut"])
        self.cutflow.SetBinContent(20-3,self.cutflow_events["associatedfatJet_has2subjets"]) 
        outputFile.mkdir("hists"); outputFile.cd("hists") ; self.cutflow.Write() ; self.argmaxAK4Jet.Write()
        for key in self.debug_vars.keys():
                for subkey in self.debug_vars[key].keys():
                        self.debug_vars[key][subkey].Write()
        #file_out.write(self.cutflow_events)
        pass
    
    def get_veto_electrons(self,lep): 
        select = False
        if(lep.pt > 10 and lep.cutBased >=1 and abs(lep.eta+lep.deltaEtaSC)<2.5): #cut-based ID Fall17 V2; only consider electrons passing veto Electron ID,pt,eta cuts
        #ele dz and dxy selection
                if( (abs(lep.eta+lep.deltaEtaSC) <= 1.479) and (abs(lep.dxy) < 0.05) and (abs(lep.dz) < 0.1 )): select = True
                elif( (abs(lep.eta+lep.deltaEtaSC) > 1.479) and (abs(lep.dxy) < 0.1) and (abs(lep.dz) < 0.2 )): select = True
        return select

    def pass_muon_CRselection(self,muons): # exactly 1 tight muon satisfying pt and eta requirements
        event_passes = False
        count_tight = 0; selected_tight_muon = []
        for mu in muons: 
                if(mu.pt > 20 and abs(mu.eta)<2.4): #tight_cand
                        if(mu.tightId and mu.pfRelIso04_all<0.15): #tight id and iso requirements
                                count_tight +=1 ; selected_tight_muon.append(mu)
        if(count_tight==1): event_passes = True              
        return [event_passes,selected_tight_muon]

    def get_veto_photons(self,ph):
        select = False
        if (ph.pt > 15 and abs(ph.eta)<2.5): 
                #Id and ele veto
                if(ph.cutBased >=1 and ph.electronVeto):   select = True      
        return select

    def get_loose_bjets(self,b_cand):
        select = False
        kinematic = (b_cand.pt > 20) and (abs(b_cand.eta)<2.4)
        bjet_tag =  b_cand.btagDeepB > 0.4184
        if(kinematic and bjet_tag): select = True
        return select

    def match_hps_lightleptonarray(self,hps_eta,hps_phi,lepton_array):
        to_return = 0 
        for i in range(len(lepton_array)):
                dR = math.sqrt((hps_eta-lepton_array[i].eta)**2+(hps_phi-lepton_array[i].phi)**2)
                if dR < 0.4 : to_return = 1
        return to_return  

    def get_clean_hps_jets(self,hpsjets,veto_electrons,veto_muons,veto_photons):
            clean_jets = []
            for hps_cand in hpsjets:
                    if(self.match_hps_lightleptonarray(hps_cand.eta,hps_cand.phi,veto_electrons)): continue
                    elif(self.match_hps_lightleptonarray(hps_cand.eta,hps_cand.phi,veto_muons)): continue
                    elif(self.match_hps_lightleptonarray(hps_cand.eta,hps_cand.phi,veto_photons)): continue
                    else: clean_jets.append(hps_cand)
            clean_jets.sort(key=lambda x: x.pt, reverse=True)        
            return clean_jets

    def get_argmax(self,hpsjets):
        max_index = -1
        pt = 0 
        for i,hps in enumerate(hpsjets):
                 if hps.pt > pt : 
                         max_index = i; pt = hps.pt
        return max_index

    def match_hps_fatjetarray(self,hps_eta,hps_phi,fatjets):
        fatjet_idx = -1 
        dR = 1000
        for i in range(len(fatjets)):
                dR_tmp = math.sqrt((hps_eta-fatjets[i].eta)**2+(hps_phi-fatjets[i].phi)**2)
                if((dR_tmp < dR) and (dR_tmp < 0.4)): dR = dR_tmp; fatjet_idx = i
        return fatjet_idx 

    def zmass_hps_lightleptonarray(self,hps_eta,hps_phi,lepton_array):
        to_return = 0
        Zmass_window = [60,120]
        indices = [x for x in range(len(lepton_array))]
        lept_pairs = list(itertools.combinations(indices,2))
        for i in range(len(lept_pairs)):
                if(to_return == 1): break # already 1, no need to check other pairs
                lept1 = lepton_array[lept_pairs[i][0]]; lept2 = lepton_array[lept_pairs[i][1]]
                z_cand = lept1.p4() + lept2.p4()
                dR = math.sqrt((hps_eta-z_cand.Eta())**2+(hps_phi-z_cand.Phi())**2)
                if( (lept1.charge*lept2.charge <0)  and  (dR < 0.4)): 
                        if(Zmass_window[0] < z_cand.M() < Zmass_window[1]): to_return =1
        return to_return

    def getMt(self,selected_tight_lep_pt,selected_tight_lep_phi,MET_pt,MET_phi):
        deltaPhi = abs(selected_tight_lep_phi-MET_phi)
        to_return = math.sqrt( 2 * selected_tight_lep_pt * MET_pt * (1 - math.cos(deltaPhi)) )
        return to_return    

    def get_debug_vars(self,CRmuon,tag_num,event_weight):
        #print("Starting to fill debug vars")    
        mu = CRmuon
        mu_tag = "muon_%s"%str(tag_num)
        if mu_tag not in self.debug_vars.keys():
                self.debug_vars[mu_tag] = {
                        "pt": ROOT.TH1F('%s_pt'%mu_tag, '%s_pt'%mu_tag, 20, 0, 1000),
                        "eta":ROOT.TH1F('%s_eta'%mu_tag, '%s_eta'%mu_tag, 50, -5, 5),
                        "phi": ROOT.TH1F('%s_phi'%mu_tag, '%s_phi'%mu_tag, 40, -4, 4)
                }
        self.debug_vars[mu_tag]["pt"].Fill(mu.pt,event_weight); self.debug_vars[mu_tag]["eta"].Fill(mu.eta,event_weight); self.debug_vars[mu_tag]["phi"].Fill(mu.phi,event_weight)
        #print("done filling debug vars")
        return

    def fill_output_branches(self,AK4Jet,recoil,selected_tight_muon,Associatedfatjet_index):
        #AK4Jet
        self.out.fillBranch("AK4Jet_pt"  ,AK4Jet.pt);  self.out.fillBranch("AK4Jet_mass"  ,AK4Jet.mass);   self.out.fillBranch("AK4Jet_nConstituents"  ,AK4Jet.nConstituents)
        self.out.fillBranch("AK4Jet_eta" ,AK4Jet.eta); self.out.fillBranch("AK4Jet_chHEF"  ,AK4Jet.chHEF); self.out.fillBranch("AK4Jet_neEmEF"  ,AK4Jet.neEmEF)
        self.out.fillBranch("AK4Jet_phi" ,AK4Jet.phi); self.out.fillBranch("AK4Jet_jetId"  ,AK4Jet.jetId); self.out.fillBranch("AK4Jet_neHEF"  ,AK4Jet.neHEF)
        self.out.fillBranch("AK4Jet_area",AK4Jet.area)
        #AssociatedFatJet
        self.out.fillBranch("Associatedfatjet_index", Associatedfatjet_index)
        # General
        self.out.fillBranch("recoil"  ,recoil)
        mu_pt,mu_eta,mu_phi = selected_tight_muon.pt,selected_tight_muon.eta,selected_tight_muon.phi
        #file_out.write("mu1_pt: %s ; mu1_eta: %s; mu1_phi:%s\n"%(mu1_pt,mu1_eta,mu1_phi))
        #file_out.write("mu2_pt: %s ; mu2_eta: %s; mu2_phi:%s\n"%(mu2_pt,mu2_eta,mu2_phi))
        self.out.fillBranch("muon_pt" , mu_pt); self.out.fillBranch("muon_eta" , mu_eta); self.out.fillBranch("muon_phi" , mu_phi)
        #file_out.write("Finished writing Muon pt,eta variables\n")
        # event weights
        if not(self.is_data):
                #for key,value in self.event_weights.items(): print("%s : %s \n"%(key,value))
                self.out.fillBranch("evt_wt_met_trig" ,self.event_weights["met_trig"]); self.out.fillBranch("evt_wt_PU" ,self.event_weights["PU"]); self.out.fillBranch("evt_wt_gen_wt" ,self.event_weights["gen_wt"]); self.out.fillBranch("evt_wt_nlo_ewk" ,self.event_weights["nlo_ewk"]); self.out.fillBranch("evt_wt_nlo_qcd_ewk" ,self.event_weights["nlo_qcd_ewk"])
                self.out.fillBranch("evt_wt_nnlo_qcd" ,self.event_weights["nnlo_qcd"]); self.out.fillBranch("evt_wt_mu_id_iso" ,self.event_weights["id_iso"]); self.out.fillBranch("evt_wt_hpsjet_sf" ,self.event_weights["hpsjet_sf"]); self.out.fillBranch("evt_wt_all" ,self.event_weights["all"])

    def analyze(self, event):
        #file_out.write("***********************************************************************************\n")    
        to_return = False; event_weight = 1   
        if not(self.is_data):  
                event_weight = self.apply_pu_weights(event,event_weight)
                event_weight = self.apply_gen_weights(event,event_weight)
                if self.processname in ["Z1Jets","Z2Jets","WJets","GJets","DY1Jets","DY2Jets"]: event_weight = self.apply_kfactors(event,event_weight)
        self.cutflow_events["total_events"] += 1*self.event_weights["gen_wt"]#1*event_weight
            
        if(not(event.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight or event.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60)) : return to_return # MET and HT Trigger 2017,2018(MonoJet AN)
        #should apply trigger sf here, but based on recoil value, so applying below after CRselection
        self.cutflow_events["passing_trigger"] += 1*event_weight

        # MET Filters
        met_filters = [event.Flag_goodVertices,event.Flag_globalSuperTightHalo2016Filter,event.Flag_HBHENoiseFilter,event.Flag_HBHENoiseIsoFilter,event.Flag_EcalDeadCellTriggerPrimitiveFilter,event.Flag_BadPFMuonFilter,event.Flag_ecalBadCalibFilterV2]
        if self.is_data: met_filters = met_filters + [event.Flag_eeBadScFilter]#; file_out.write("$$$$$ DATA!!!!$$$$$")
        if not(all(filter_flag == 1 for filter_flag in met_filters)): return to_return
        self.cutflow_events["MET_filters"] += 1*event_weight
        
        dphi_PFMET_TkMET = abs(event.MET_phi-event.TkMET_phi)
        if not(dphi_PFMET_TkMET < 2): return to_return
        self.cutflow_events["dphi_PFMET_TkMET"] += 1*event_weight

        dMET_PF_Calo = abs((event.MET_pt-event.CaloMET_pt)/event.MET_pt)
        if not(dMET_PF_Calo < 0.5): return to_return
        self.cutflow_events["dMET_PF_Calo"] += 1*event_weight   

        muons = Collection(event, "Muon")
        pass_CR_selection,selected_tight_muon_list = self.pass_muon_CRselection(muons)
        if not(pass_CR_selection): return to_return
        selected_tight_muon = selected_tight_muon_list[0] # only 1 tight muon
        met_4vec = ROOT.TLorentzVector()
        met_4vec.SetPtEtaPhiE(event.MET_pt,0,event.MET_phi,event.MET_pt)# since Met itself is "transverse plane", take eta =0
        event.recoil = (selected_tight_muon.p4() + met_4vec).Pt() # vector sum
        if not(self.is_data):  
                event_weight = self.apply_trigger_sf(event,event_weight)
                event_weight = self.apply_id_iso_weights(event,event_weight,selected_tight_muon)
        # fixing the previous cutflow values because recoil wasn't available earlier FIXME
        self.cutflow_events["muon_CRselection"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,6,event_weight)

        lepMET_mt = self.getMt(selected_tight_muon.pt,selected_tight_muon.phi,event.MET_pt,event.MET_phi)
        if (lepMET_mt >= 160): return to_return
        self.cutflow_events["lepMET_MtCut"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,7,event_weight)

        electrons = Collection(event, "Electron")
        veto_electrons = filter(self.get_veto_electrons, electrons) 
        if (len(veto_electrons)>0): return to_return 
        self.cutflow_events["electron_veto"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,8,event_weight)

        photons = Collection(event, "Photon")
        veto_photons = filter(self.get_veto_photons, photons)
        if (len(veto_photons)>0): return to_return 
        self.cutflow_events["photon_veto"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,9,event_weight)

        jets = Collection(event, "Jet")
        #loose_bjets = filter(self.get_loose_bjets, jets)
        #if (len(loose_bjets)>0): return to_return 
        #self.cutflow_events["bjet_veto"] += 1*event_weight; self.get_debug_vars(selected_muons,10,event_weight)

        #hpsjets = Collection(event, "Tau")
        if(len(jets)<=0): return to_return
        # can't apply weight here because the hps object is not yet selected, so applying it below
        self.cutflow_events["atleast_1_ak4jet"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,11-1,event_weight)

        clean_ak4_jets = self.get_clean_hps_jets(jets,veto_electrons,selected_tight_muon_list,veto_photons)
        if(len(clean_ak4_jets)<=0): return to_return 
        self.cutflow_events["atleast_1_cleanak4jet"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,12-1,event_weight)        
        AK4Jet = clean_ak4_jets[0] #Choosing highest Pt clean ak4Jet 
        max_index = self.get_argmax(clean_ak4_jets)
        #if not(max_index ==0):file_out.write("HPS max pt index = %s"%max_index)
        self.argmaxAK4Jet.Fill(max_index)     
        #if not(self.is_data):  event_weight = self.apply_hpsjet_sf(AK4Jet,event_weight)

        dphi_AK4Jet_MET = abs(AK4Jet.phi-event.MET_phi)
        if not(dphi_AK4Jet_MET > 0.5 ): return to_return
        self.cutflow_events["dphi_AK4Jet_MET"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,13-1,event_weight)

        #HPSJet cleaning
        if not((AK4Jet.eta < 2.4) and (AK4Jet.pt > 100)): return to_return
        self.cutflow_events["AK4Jet_ptetacuts"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,14-1,event_weight)
        #if (self.match_hps_lightleptonarray(HPSJet.eta,HPSJet.phi,veto_electrons) or self.match_hps_lightleptonarray(HPSJet.eta,HPSJet.phi,selected_muons)): return to_return #WJets cleaning
        #self.cutflow_events["wjets_cleaning"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,15-1,event_weight)
        #if (self.zmass_hps_lightleptonarray(HPSJet.eta,HPSJet.phi,veto_electrons) or self.zmass_hps_lightleptonarray(HPSJet.eta,HPSJet.phi,selected_muons)): return to_return #Z(ll)Jets cleaning
        #self.cutflow_events["zjets_cleaning"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,16-1,event_weight)   

        #HPSJet_jetIdx = HPSJet.jetIdx
        #if not(0<=HPSJet_jetIdx<len(jets)):  return to_return
        #Associatedak4Jet = jets[HPSJet_jetIdx]
        if(AK4Jet.jetId <6):  return to_return #Jet ID flags bit1 is loose (always false in 2017 since it does not exist), bit2 is tight, bit3 is tightLepVeto
        self.cutflow_events["AK4Jet_tightLepVeto"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,17-3,event_weight) 

        fatjets = Collection(event, "FatJet")
        fatjet_idx = self.match_hps_fatjetarray(AK4Jet.eta,AK4Jet.phi,fatjets)# HPSJet eta and phi are for a single HPSJet, fatjets is collection  
        if not(0<=fatjet_idx<len(fatjets)): return to_return
        AssociatedFatJet = fatjets[fatjet_idx]
        self.cutflow_events["has_associatedfatJet"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,18-3,event_weight)

        if(event.recoil <= 250): return to_return
        self.cutflow_events["recoil_cut"] += 1*event_weight; self.get_debug_vars(selected_tight_muon,19-3,event_weight)

        #file_out.write(" Getting event weight!\n")
        #if not(self.is_data): self.event_weights["all"] = self.get_event_weight(event,selected_muons)
        if not(self.is_data): self.event_weights["all"] = event_weight
        #file_out.write("event weight =%s\n"%self.event_weights["all"])
        self.fill_output_branches(AK4Jet,event.recoil,selected_tight_muon,fatjet_idx)

        subjets = Collection(event, "SubJet")
        fatjet_subjet_IDx1 = AssociatedFatJet.subJetIdx1 # index of first subjet of AssociatedFatJet
        fatjet_subjet_IDx2 = AssociatedFatJet.subJetIdx2 # index of second subjet of AssociatedFatJet
        if((0<=fatjet_subjet_IDx1<len(subjets)) and (0<=fatjet_subjet_IDx2<len(subjets))):
                self.cutflow_events["associatedfatJet_has2subjets"] += 1*event_weight
        subjet1 = subjets[fatjet_subjet_IDx1]; subjet2 =  subjets[fatjet_subjet_IDx2] 

        #self.fill_output_branches(HPSJet,Associatedak4Jet,AssociatedFatJet,recoil,subjet1,subjet2)      

        to_return = True 
        #file_out.write("***********************************************************************************\n")
        return to_return

parser = argparse.ArgumentParser(
    "Analyzing the nanoAOD tau-objects(called HPS Jets) and associated jets for SingleMuCR")
parser.add_argument(
    "inputnanoAODFile",
    action="store",
    nargs='?',
    default = "/hdfs/store/user/mallampalli/analysis/Mono-zprime/nanoaod_tests/bkg_samples/DY2Jets/DY2JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8/00A4DD69-CE32-1243-AAA6-094A67EE7C4B.root",
    help="Provide the relative path to the target input nanoAOD file for signal or background")
parser.add_argument(
    "postfix_outputFilename",
    action="store",
    nargs='?',
    default = "_Skim",
    help="outputfilename = inputfilename.strip('.root')+postfix+.root")          
parser.add_argument(
    "--isdata",
    action="store",
    nargs='?',
    default = 'False',
    help="set == True for data")
parser.add_argument(
    "--era",
    action="store",
    nargs='?',
    default = 'False',
    help="set == era for data(i.e. A,B,C,D) and == None for MC")
parser.add_argument(
    "--processname",
    action="store",
    nargs='?',
    default = "DY2Jets",
    help="process name")
parser.add_argument(
    "--year",
    action="store",
    nargs='?',
    default = "2018",
    help="year")

args = parser.parse_args()
if(args.isdata == 'False'): args.isdata =  False # non-0 length evaluates to True!
elif(args.isdata == 'True'): args.isdata =  True
preselection = None#"MET_pt > 250" #don't apply preselction as we need to scale with total_events!
branchsel_in = "./keep_branches_input.txt"
branchsel_out = "./keep_branches_output_noHPS.txt"
# Get environment variables set by farmout
#'''
inputFile = [os.getenv('INPUT')]
with open(inputFile[0]) as f:
        files = f.readlines()
f.close()
files = map(lambda s: s.strip(), files)
#'''
#file_out.write("****%s***"%files)   
#files = [args.inputnanoAODFile]
if(args.isdata): 
        datapath  = os.path.join(os.environ.get('CMSSW_BASE','CMSSW_BASE'),"src/PhysicsTools/NanoAODTools/python/postprocessing/data/good_lumis/")
        jsonInput= datapath+"Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"
else: jsonInput=None

#Set jesSys to "All" for split JES systematics and "Total" for combined JES systematics
ak4JetAndMetCorrections2018 = createJMECorrector(isMC=not(args.isdata), dataYear=2018, runPeriod=args.era, jetType = "AK4PFchs", jesUncert="Total", applyHEMfix=True)
ak8JetCorrections2018 = createJMECorrector(isMC=not(args.isdata), dataYear=2018, runPeriod=args.era, jetType = "AK8PFPuppi", jesUncert="Total", applyHEMfix=True)

p = PostProcessor(".", files, prefetch=True, cut=preselection, branchsel=branchsel_in, outputbranchsel= branchsel_out, modules=[
                  HPSJetCorrs2018ReReco(),SingleMuCR_noHPS(args.isdata,args.processname,args.year),ak4JetAndMetCorrections2018(),ak8JetCorrections2018()], postfix=args.postfix_outputFilename,noOut=False,maxEntries=None,jsonInput=jsonInput)
p.run()
