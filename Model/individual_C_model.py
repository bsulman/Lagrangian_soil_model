from pylab import *


class lagrangian_soil_sim:
    
    def __init__(self,pore_distribution=array([0.3,0.4,0.3]),nlocations=200,nparticles=100,ntimes=5000):
        self.Ctype_key={'enzyme':6,'lignin':5,'insoluble polymer':4,'soluble polymer':3,'monomer':2,'microbe':1,'CO2':0}
        self.Ctype_key_inv=self.Ctype_key.__class__(map(reversed,self.Ctype_key.items()))
        self.plotcolors={'lignin':'C0','insoluble polymer':'C1','soluble polymer':'C2','monomer':'C3','microbe':'C4','CO2':'C5','enzyme':'C6'}
        
        self.pore_key={'macropore':0,'micropore':1,'nanopore':2}
        self.pore_distribution=pore_distribution  #array([0.3,0.4,0.3])

        self.location_map=zeros(nlocations,dtype=int)

        # Need to fix this
        self.location_map[0:int(pore_distribution[0]*nlocations)]=self.pore_key['macropore']
        self.location_map[int(pore_distribution[0]*nlocations):int((pore_distribution[1]+pore_distribution[0])*nlocations)]=self.pore_key['micropore']
        self.location_map[int((pore_distribution[1]+pore_distribution[0])*nlocations):]=self.pore_key['nanopore']

        self.particle_age=ma.masked_all((nparticles,ntimes),dtype=int)
        self.particle_form=ma.masked_all((nparticles,ntimes),dtype=int)
        self.particle_location=ma.masked_all((nparticles,ntimes),dtype=int)
        
        self.transform_matrix=zeros((len(self.Ctype_key),len(self.Ctype_key)))
        self.transform_matrix[self.Ctype_key['lignin'],self.Ctype_key['soluble polymer']]=1e-2
        self.transform_matrix[self.Ctype_key['insoluble polymer'],self.Ctype_key['soluble polymer']]=1e-2
        self.transform_matrix[self.Ctype_key['insoluble polymer'],self.Ctype_key['monomer']]=1e-2
        self.transform_matrix[self.Ctype_key['soluble polymer'],self.Ctype_key['monomer']]=2e-2
        self.transform_matrix[self.Ctype_key['monomer'],self.Ctype_key['CO2']]=1e-1
        
        self.prob_leaving=array([0.1,0.05,0.005])
        
        self.move_probs = {
            'lignin':array([0.01,0.01,0.001]),
            'insoluble polymer':array([0.01,0.01,0.001]),
            'soluble polymer':array([0.1,0.05,0.05]),
            'monomer':array([0.1,0.1,0.1]),
            'microbe':array([0.1,0.01,0.0]),
            'enzyme':array([0.1,0.05,0.05]),
            }
            
        self.move_probs_matrix=zeros((len(self.Ctype_key),len(self.pore_key)))
        self.move_probs_matrix[self.Ctype_key['lignin'],:]=array([0.01,0.01,0.001])
        self.move_probs_matrix[self.Ctype_key['insoluble polymer'],:]=array([0.01,0.01,0.001])
        self.move_probs_matrix[self.Ctype_key['soluble polymer'],:]=array([0.1,0.05,0.05])
        self.move_probs_matrix[self.Ctype_key['monomer'],:]=array([0.1,0.1,0.1])
        self.move_probs_matrix[self.Ctype_key['microbe'],:]=array([0.1,0.01,0.0])
        self.move_probs_matrix[self.Ctype_key['enzyme'],:]=array([0.1,0.05,0.05])
        
        return 

    def add_particle(self,pnumber,time,pore_type='macropore',C_type='insoluble polymer'):
        locs=nonzero(self.location_map==self.pore_key[pore_type])[0]
        self.particle_location[pnumber,time]=locs[randint(len(locs))]
        self.particle_age[pnumber,time]=0
        self.particle_form[pnumber,time]=self.Ctype_key[C_type]



    def transform_particles(self,time,print_transforms=True):
        microbe_pores = self.particle_location[self.particle_form[:,time]==self.Ctype_key['microbe'],time]
        enzyme_pores  = self.particle_location[self.particle_form[:,time]==self.Ctype_key['enzyme'],time] 
        
        transform_probs = zeros((len(self.particle_location),len(self.Ctype_key)))
        for micpore in microbe_pores:
            transform_probs[self.particle_location[:,time]==micpore,:]+=self.transform_matrix[self.particle_form[self.particle_location[:,time]==micpore,time],:]
        for enzpore in enzyme_pores:
            transform_probs[self.particle_location[:,time]==enzpore,:]+=self.transform_matrix[self.particle_form[self.particle_location[:,time]==enzpore,time],:]
        
        diceroll = transform_probs - random_sample(transform_probs.shape)
        transformed = diceroll.max(axis=1)>0
        transformed_to_type = diceroll.argmax(axis=1)
        
        if print_transforms:
            for transnum in nonzero(transformed)[0]:
                print('Transformation! {old:s} --> {new:s}, prob={prob:1.2f}, pore={pore:d}'.format(old=self.Ctype_key_inv[self.particle_form[transnum,time]],
                                        new=self.Ctype_key_inv[transformed_to_type[transnum]],prob=transform_probs[transnum,transformed_to_type[transnum]],pore=self.particle_location[transnum,time]))
        
        newtype = self.particle_form[:,time]
        newtype[transformed]=transformed_to_type[transformed]

        return newtype



    def move_particles(self,time):
        # Calculate movement probabilities for all particles
        moveprobs=self.move_probs_matrix[self.particle_form[:,time]]*self.prob_leaving[self.location_map[self.particle_location[:,time]],None]*self.pore_distribution[self.location_map[self.particle_location[:,time]],None]
        diceroll = moveprobs - random_sample(moveprobs.shape)
        moved = diceroll.max(axis=1)>0
        moved_to_type = diceroll.argmax(axis=1)
        destination = self.particle_location[:,time]
        for movenum in nonzero(moved)[0]:
            locs=nonzero(self.location_map==moved_to_type[movenum])[0]
            destination[movenum]=locs[randint(len(locs))]
        return destination
        
        
    def step(self,t0,t1):
        isCO2=self.particle_form[:,t0]==self.Ctype_key['CO2']
        self.particle_location[:,t1]=self.move_particles(t0)
        self.particle_age[~isCO2,t1]=self.particle_age[~isCO2,t0]+1
        self.particle_age[isCO2,t1]=self.particle_age[isCO2,t0]
        self.particle_form[:,t1]=self.transform_particles(t0)
        
    def plot_particle_cascade(self):
        for pnum in range(len(self.particle_form)):
            form=self.particle_form[pnum,:]
            location=self.particle_location[pnum,:]
            location_type=self.location_map[location]
            age=self.particle_age[pnum,:]

            pore_lines=['-','--',':']
            offset=rand()*0.2
            c='C%d'%(pnum%10)
            for loctype in self.pore_key.values():
                plot(age,ma.masked_array(form,mask=~(location_type==loctype)|(form==self.Ctype_key['microbe']))+offset,ls=pore_lines[loctype],c=c)

        yticks(list(self.Ctype_key.values()),list(self.Ctype_key.keys()))
        ylim(0.2,5.5)

        legend([Line2D([0],[0],ls='-'),Line2D([0],[0],ls='--'),Line2D([0],[0],ls=':')],['macropore','micropore','nanopore'])

        xlabel('Particle age')
        title('History of particle C type')


    def plot_particle_loc_history(self):
        for pnum in range(len(self.particle_form)):
            form=self.particle_form[pnum,:]
            location=self.particle_location[pnum,:]
            location_type=self.location_map[location]
            age=self.particle_age[pnum,:]

            offset=rand()*0.2
            for Ctype in self.Ctype_key:
                if Ctype=='CO2':
                    continue
                elif Ctype=='microbe':
                    plot(age,ma.masked_array(location,mask=~(form==self.Ctype_key[Ctype]))+offset,marker='s',ls='',ms=0.5,c=self.plotcolors[Ctype],alpha=0.5,zorder=0)
                else:
                    plot(age,ma.masked_array(location,mask=~(form==self.Ctype_key[Ctype]))+offset,marker='.',ls='',ms=0.5,c=self.plotcolors[Ctype])


            transformations=nonzero(diff(form))[0]
            for trans in transformations:
                if form[trans+1]==self.Ctype_key['CO2']:
                    scatter(age[trans],location[trans],marker='x',s=30.0,facecolor=self.plotcolors[self.Ctype_key_inv[form[trans]]],zorder=100)
                else:
                    scatter(age[trans],location[trans],marker='o',s=20.0,edgecolor=self.plotcolors[self.Ctype_key_inv[form[trans]]],facecolor='None',zorder=100)

        
        boundaries=nonzero(diff(self.location_map)!=0)[0]
        poresizes=['Macropore','Micropore','Nanopore']
        text(age[-1],0+2.5,poresizes[self.location_map[0]],rotation=90,va='bottom')
        for num in boundaries:
            plot(age,zeros(len(age))+num+0.5,'k--',lw=2)
            text(age[-1],num+2.5,poresizes[self.location_map[num+1]],rotation=90,va='bottom')
        for num in range(len(self.location_map)):
            plot(age,zeros(len(age))+num+0.5,'k--',lw=0.2)

        Ctypes=['lignin','insoluble polymer','soluble polymer','monomer','microbe']
        legend([Line2D([0],[0],ls='',marker='.',c=self.plotcolors[Ctype]) for Ctype in Ctypes],Ctypes,loc=(0.0,1.06),ncol=3)

        xlabel('Time')
        ylabel('Pore location')
        title('History of particle location')
        
        
        
    def plot_histogram(self,separate_pores=False):
        age=arange(self.particle_age.shape[1])
        if separate_pores:
            bottom=zeros(self.particle_location.shape[1])
            location_type=ma.masked_array(self.location_map[self.particle_location],mask=self.particle_location.mask)
            old_bottom=bottom[-1]
            for poresize in ['macropore','micropore','nanopore']:
                for Ctype in ['lignin','insoluble polymer','soluble polymer','monomer','microbe']:
                    top=bottom+((self.particle_form==self.Ctype_key[Ctype])&(location_type==self.pore_key[poresize])).sum(axis=0)/self.particle_form.count(axis=0)
                    fill_between(age,bottom,top,label=Ctype,color=self.plotcolors[Ctype])
                    bottom=top
                plot(age,top,'k-',lw=2.0)
                text(age[-1],old_bottom,poresize[:-4],rotation=60,va='bottom')
                old_bottom=bottom[-1]


            xlabel('Time')
            ylabel('Relative amount')
            title('Relative amount of each particle type in each pore size')

            
            
        else:
            bottom=zeros(self.particle_form.shape[1])
            for Ctype in ['microbe','lignin','insoluble polymer','soluble polymer','monomer','CO2']:
                top=bottom+(self.particle_form==self.Ctype_key[Ctype]).sum(axis=0)/self.particle_form.count(axis=0)
                fill_between(age,bottom,top,label=Ctype,color=self.plotcolors[Ctype])
                bottom=top

            legend(ncol=2)
            xlabel('Time')
            ylabel('Relative amount')
            title('Relative total amount of each C type over time')




###########################################
###########   Run sims  ###################
###########################################


# Initial conditions


total_particles=0

# Initialize a simulation
nparticles=100
nlocations=200
ntimes=5000
sim = lagrangian_soil_sim(nparticles=nparticles,nlocations=nlocations)
sim_mobile = lagrangian_soil_sim(nparticles=nparticles,nlocations=nlocations)
sim_mobile.move_probs_matrix = sim_mobile.move_probs_matrix*2

# Add some microbes
nmicrobes=5
for ii in range(nmicrobes):
    sim.add_particle(ii,0,C_type='microbe')
    sim_mobile.add_particle(ii,0,C_type='microbe')
    total_particles += 1
for ii  in range(nparticles-nmicrobes):
    sim.add_particle(total_particles,0,C_type=['lignin','insoluble polymer','soluble polymer'][randint(3)])
    sim_mobile.add_particle(total_particles,0,C_type=['lignin','insoluble polymer','soluble polymer'][randint(3)])
    total_particles += 1

# Iterate model
# Seed the pseudo-random number generator, for reproducible simulations
np.random.seed(1)
for tt in range(1,ntimes):
    if tt%100==0:
        print(tt)
        
    sim.step(tt-1,tt)

np.random.seed(1)
for tt in range(1,ntimes):
    if tt%100==0:
        print(tt)
        
    sim_mobile.step(tt-1,tt)



###########################################
###########   Plots  ######################
###########################################

histfig=figure('Particle histories',figsize=(13,6))
histfig.clf()

subplot(121)
sim.plot_particle_cascade()

subplot(122)
sim.plot_particle_loc_history()

tight_layout()


histogramfig=figure('Histograms',figsize=(8.5,8))
histogramfig.clf()
subplot(211)
sim.plot_histogram(separate_pores=False)

subplot(212)
sim.plot_histogram(separate_pores=True)
tight_layout()


histfig=figure('Particle histories (mobile)',figsize=(13,6))
histfig.clf()

subplot(121)
sim_mobile.plot_particle_cascade()

subplot(122)
sim_mobile.plot_particle_loc_history()

tight_layout()


histogramfig=figure('Histograms (mobile)',figsize=(8.5,8))
histogramfig.clf()
subplot(211)
sim_mobile.plot_histogram(separate_pores=False)

subplot(212)
sim_mobile.plot_histogram(separate_pores=True)
tight_layout()


show()
