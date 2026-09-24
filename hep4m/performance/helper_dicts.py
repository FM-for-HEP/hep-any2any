### see https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf


def get_class_mass_dict(n_classes=3):
    if n_classes == 3:

        ###  0: charged particles
        ###  1: neutral hadrons
        ###  2: photons
        class_mass_dict = {
            0: 0.2760, # ch had
            1: 0.76419, # neut had
            2: 0.0, # gamma
        }

    elif n_classes == 5:

        ###  0: charged hadrons
        ###  1: electrons      
        ###  2: muons          
        ###  3: neutral hadrons
        ###  4: photons    
        ###  5: residual
        ### -1: neutrinos
        class_mass_dict = {
            0: 0.2760, # ch had
            1: 0.00051, # e
            2: 0.10566, # mu
            3: 0.76419, # neut had
            4: 0.0, # gamma
            5: 0.0 # residual
        }

    return class_mass_dict