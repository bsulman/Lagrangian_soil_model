from pylab import *


class lagrangian_soil_sim:
    
    def __init__(self,pore_distribution=array([0.3,0.4,0.3]),nlocations=200,nparticles=100,dt=4/(24*365)):
        self.Ctypes=['CO2','microbe','monomer','soluble polymer','insoluble polymer','lignin','enzyme']
        # self.plotcolors={'lignin':'C0','insoluble polymer':'C1','soluble polymer':'C2','monomer':'C3','microbe':'C4','CO2':'C5','enzyme':'C6'}
        self.plotcolors={'lignin':[0,0,0],'insoluble polymer':[0.4,0.4,0.4],'soluble polymer':[0.7,0.7,0.7],'monomer':[1.0,1.0,1.0],'microbe':[1.0,1.0,1.0],'CO2':[0.8,0.8,0.8],'enzyme':'C6'}
        self.hatching={'microbe':'|','CO2':'//'}
        self.dt=dt
        
        self.poretypes=['macropore','micropore','nanopore']
        self.pore_distribution=pore_distribution  #array([0.3,0.4,0.3])

        self.location_map=zeros(nlocations,dtype=int)

        # Need to fix this
        self.location_map[0:int(pore_distribution[0]*nlocations)]=self.poretypes.index('macropore')
        self.location_map[int(pore_distribution[0]*nlocations):int((pore_distribution[1]+pore_distribution[0])*nlocations)]=self.poretypes.index('micropore')
        self.location_map[int((pore_distribution[1]+pore_distribution[0])*nlocations):]=self.poretypes.index('nanopore')

        self.particle_age=zeros((nparticles,),dtype=int)
        self.particle_form=zeros((nparticles,),dtype=int)
        self.particle_location=zeros((nparticles,),dtype=int)
        
        self.transform_matrix=zeros((len(self.Ctypes),len(self.Ctypes)))
        self.transform_matrix[self.Ctypes.index('lignin'),self.Ctypes.index('soluble polymer')]=1e-2
        self.transform_matrix[self.Ctypes.index('insoluble polymer'),self.Ctypes.index('soluble polymer')]=1e-2
        self.transform_matrix[self.Ctypes.index('insoluble polymer'),self.Ctypes.index('monomer')]=1e-2
        self.transform_matrix[self.Ctypes.index('soluble polymer'),self.Ctypes.index('monomer')]=2e-2
        self.transform_matrix[self.Ctypes.index('monomer'),self.Ctypes.index('CO2')]=1e-1
        
        self.prob_leaving=array([0.05,0.005,0.0005])
        
        self.move_probs = {
            'lignin':array([0.01,0.01,0.001]),
            'insoluble polymer':array([0.01,0.01,0.001]),
            'soluble polymer':array([0.1,0.05,0.05]),
            'monomer':array([0.1,0.1,0.1]),
            'microbe':array([0.1,0.01,0.0]),
            'enzyme':array([0.1,0.05,0.05]),
            }
            
        self.move_probs_matrix=zeros((len(self.Ctypes),len(self.poretypes)))
        self.move_probs_matrix[self.Ctypes.index('lignin'),:]=array([0.01,0.01,0.001])
        self.move_probs_matrix[self.Ctypes.index('insoluble polymer'),:]=array([0.01,0.01,0.001])
        self.move_probs_matrix[self.Ctypes.index('soluble polymer'),:]=array([0.1,0.05,0.05])
        self.move_probs_matrix[self.Ctypes.index('monomer'),:]=array([0.1,0.1,0.1])
        self.move_probs_matrix[self.Ctypes.index('microbe'),:]=array([0.1,0.001,0.0])
        self.move_probs_matrix[self.Ctypes.index('enzyme'),:]=array([0.1,0.05,0.05])
        
        self.move_hist=[]
        self.transform_hist=[]
        
        return 

    def add_particle(self,pnumber,pore_type='macropore',C_type='insoluble polymer'):
        if pore_type=='random':
            xx=self.move_probs_matrix[self.Ctypes.index(C_type),:]*self.pore_distribution
            dist=(xx/xx.sum()).cumsum() 
            x=rand()
            pore_type=self.poretypes[nonzero(x<dist)[0][0]]

        locs=nonzero(self.location_map==self.poretypes.index(pore_type))[0]
        self.particle_location[pnumber]=locs[randint(len(locs))]
        self.particle_age[pnumber]=0
        self.particle_form[pnumber]=self.Ctypes.index(C_type)
        
        self.move_hist.append({'particle':pnumber,'time':0,'from':-999,'to':self.particle_location[pnumber]})
        self.transform_hist.append({'particle':pnumber,'time':0,'from':-999,'to':self.particle_form[pnumber]})


    def transform_particles(self,print_transforms=True):
        microbe_pores = self.particle_location[self.particle_form==self.Ctypes.index('microbe')]
        enzyme_pores  = self.particle_location[self.particle_form==self.Ctypes.index('enzyme')] 
        
        transform_probs = zeros((len(self.particle_location),len(self.Ctypes)))
        for micpore in microbe_pores:
            transform_probs[self.particle_location==micpore]+=self.transform_matrix[self.particle_form[self.particle_location==micpore]]
        for enzpore in enzyme_pores:
            transform_probs[self.particle_location==enzpore]+=self.transform_matrix[self.particle_form[self.particle_location==enzpore]]
        
        diceroll = transform_probs - random_sample(transform_probs.shape)
        transformed = diceroll.max(axis=1)>0
        transformed_to_type = diceroll.argmax(axis=1)
        
        if print_transforms:
            for transnum in nonzero(transformed)[0]:
                print('Transformation! {old:s} --> {new:s}, prob={prob:1.2f}, pore={pore:d}'.format(old=self.Ctypes[self.particle_form[transnum]],
                                        new=self.Ctypes[transformed_to_type[transnum]],prob=transform_probs[transnum,transformed_to_type[transnum]],pore=self.particle_location[transnum]))
        
        newtype = self.particle_form.copy()
        newtype[transformed]=transformed_to_type[transformed]

        return newtype



    def move_particles(self):
        # Calculate movement probabilities for all particles
        moveprobs=self.move_probs_matrix[self.particle_form]*self.pore_distribution*self.prob_leaving[self.location_map[self.particle_location],None]
        diceroll = moveprobs - random_sample(moveprobs.shape)
        moved = diceroll.max(axis=1)>0
        moved_to_type = diceroll.argmax(axis=1)
        destination = self.particle_location.copy()
        for movenum in nonzero(moved)[0]:
            locs=nonzero(self.location_map==moved_to_type[movenum])[0]
            destination[movenum]=locs[randint(len(locs))]
        return destination
        
        
    def step(self):
        newlocs=self.move_particles()
        moved=nonzero(newlocs != self.particle_location)[0]
        for p in moved:
            self.move_hist.append({'particle':p,'time':self.particle_age[p],'from':self.particle_location[p],'to':newlocs[p]})
        self.particle_location=newlocs
        
        isCO2=self.particle_form==self.Ctypes.index('CO2')
        self.particle_age[~isCO2]=self.particle_age[~isCO2]+1
        self.particle_age[isCO2]=self.particle_age[isCO2]
        newforms=self.transform_particles()
        changed=nonzero(newforms != self.particle_form)[0]
        for p in changed:
            self.transform_hist.append({'particle':p,'time':self.particle_age[p],'from':self.particle_form[p],'to':newforms[p]})
        self.particle_form=newforms
        
    def get_hist_array(self,hist,pnum,tstep=None,length=None):
        if tstep is None:
            tstep=self.dt
        if pnum>len(self.particle_form):
            raise valueError('pnum is greater than number of particles')
        hist_p=[m for m in hist if m['particle']==pnum]
        if length is None:
            length=self.particle_age.max()/tstep
        out=zeros(int(length),dtype=int)
        for x in range(len(hist_p)-1):
            start=int(hist_p[x]['time']*self.dt/tstep)
            stop=int(hist_p[x+1]['time']*self.dt/tstep)
            out[start:stop]=hist_p[x]['to']
        out[int(hist_p[-1]['time']//tstep):]=hist_p[-1]['to']
        
        return hist_p,out
        
        
def plot_particle_cascade(self,start=0,end=None):
    for pnum in range(len(self.particle_form)):
        age=arange(0,self.particle_age[pnum]/self.dt)[start:end]*self.dt
        t,form=self.get_hist_array(self.transform_hist,pnum,length=len(age))
        form=form[start:end]
        m,location=self.get_hist_array(self.move_hist,pnum,length=len(age))
        location=location[start:end]
        location_type=self.location_map[location]
        

        pore_lines=['-','--',':']
        offset=rand()*0.2
        c='C%d'%(pnum%10)
        for loctype in range(len(self.poretypes)):
            plot(age,ma.masked_array(form,mask=~(location_type==loctype)|(form==self.Ctypes.index('microbe')))+offset,ls=pore_lines[loctype],c=c)

    yticks(arange(len(self.Ctypes)),self.Ctypes)
    ylim(0.2,5.5)

    legend([Line2D([0],[0],ls='-'),Line2D([0],[0],ls='--'),Line2D([0],[0],ls=':')],['macropore','micropore','nanopore'])

    xlabel('Particle age')
    title('History of particle C type')


def plot_particle_loc_history(self,do_legend=False,dt=1,particles=None,draw_pores=True,start=0,end=None):
    start=int(start)
    if end is not None:
        end=int(end)
    if particles is None:
        particles=arange(len(self.particle_form))
    for pnum in particles:
        transforms,form=self.get_hist_array(self.transform_hist,pnum,tstep=dt)
        moves,location=self.get_hist_array(self.move_hist,pnum,tstep=dt)
        location_type=self.location_map[location]
        age=arange(0,self.particle_age.max(),dt)[start:end]
        form=form[start:end]
        location=location[start:end]

        # movements=nonzero(diff(location))[0]
        # transformations=nonzero(diff(form))[0]
        movements=array([m['time']*self.dt/dt for m in moves[1:]],dtype=int)
        movements=movements[movements<end]
        transformations=array([t['time']*self.dt/dt for t in transforms[1:]],dtype=int)
        transformations=transformations[transformations<end]
        bounds=sort(concatenate(([0],movements,transformations,[len(form)-1])))
        offset=rand()*0.2
        for num in range(len(bounds)-1):
            Ctype=self.Ctypes[form[bounds[num]]]
            if Ctype=='CO2':
                continue
            r=Rectangle(xy=(age[bounds[num]],location[bounds[num]]+offset),width=age[bounds[num+1]]-age[bounds[num]],height=2.5,
                facecolor=self.plotcolors[Ctype],hatch=self.hatching.get(Ctype,None),edgecolor='k')
            gca().add_patch(r)
        # for Ctype in self.Ctype_key:
        #     if Ctype=='CO2':
        #         continue
        #     elif Ctype=='microbe':
        #         plot(age,ma.masked_array(location,mask=~(form==self.Ctype_key[Ctype]))+offset,marker='s',ls='',ms=0.5,c=self.plotcolors[Ctype],alpha=0.5,zorder=0)
        #     else:
        #         plot(age,ma.masked_array(location,mask=~(form==self.Ctype_key[Ctype]))+offset,marker='.',ls='',ms=0.5,c=self.plotcolors[Ctype])


        for trans in transformations:
            if form[trans]==self.Ctypes.index('CO2'):
                scatter(age[max(0,trans-1)],location[max(0,trans-1)],marker='x',s=30.0,facecolor=self.plotcolors[self.Ctypes[form[max(0,trans-1)]]],zorder=100)
            else:
                scatter(age[max(0,trans-1)],location[max(0,trans-1)],marker='o',s=20.0,edgecolor=self.plotcolors[self.Ctypes[form[max(0,trans-1)]]],facecolor='None',zorder=100)

        for mov in movements:
            annotate('',xytext=(age[max(0,mov-1)],location[max(0,mov-1)]),xy=(age[mov],location[mov]),color=self.plotcolors[self.Ctypes[form[mov]]],
                        arrowprops={'arrowstyle':'->'})
    
    boundaries=nonzero(diff(self.location_map)!=0)[0]
    poresizes=['Macropore','Micropore','Nanopore']
    # text(age[-1],0+2.5,poresizes[self.location_map[0]],rotation=60,va='bottom',fontsize='small')
    b=concatenate(([0],boundaries,[len(self.location_map)-1]))
    for num in range(len(b)-1):
        annotate(poresizes[num][:-4].capitalize(),xytext=(age[-1]*1.03,(b[num]+b[num+1])*0.5),xy=(age[-1]*1.01,(b[num]+b[num+1])*0.5),
                    arrowprops={'arrowstyle':'-[, widthB=%1.1f, lengthB=0.4'%(abs(b[num]-b[num+1])*0.5e-1)},va='center')
    p=Polygon(array([[age[-1]*1.15,0],[age[-1]*1.3,0],[age[-1]*1.225,len(self.location_map)]]),facecolor=[0.25,0.25,0.25])
    gca().add_patch(p)
    text(age[-1]*1.225,len(self.location_map)/2,'Pore size class',color='w',ha='center',va='center',rotation=90)
    for num in boundaries:
        plot(age,zeros(len(age))+num+0.5,'k--',lw=2)
        # text(age[-1],num+2.5,poresizes[self.location_map[num+1]],rotation=60,va='bottom',fontsize='small')
    if draw_pores:
        for num in range(0,len(self.location_map),4):
            plot([age[0],age[-1]],zeros(2)+num,'k--',lw=0.2,color=[0.5,0.5,0.5])

    Ctypes=['lignin','insoluble polymer','soluble polymer','monomer','microbe']
    if do_legend:
        legend([Rectangle(xy=(0,0),width=0,height=2.5,
            facecolor=self.plotcolors[Ctype],hatch=self.hatching.get(Ctype,None),edgecolor='k') for Ctype in Ctypes]+[Line2D([0],[0],marker='o',ls='None',markeredgecolor='k',markerfacecolor='w'),Line2D([0],[0],marker='x',ls='None',color='k')],
            [c.capitalize() for c in Ctypes]+['C transformation','CO$_2$ produced'],loc=(0.0,1.08),ncol=3)

    xlabel('Time (years)',fontsize='large')
    ylabel('Individual pore spaces',fontsize='large')
    yticks(arange(0,len(self.location_map),4),labels=[])
    title('History of particle location')
    ylim(0,len(self.location_map)+1)
    gca().spines['right'].set_visible(False)
    xlim(0,age[-1]*1.3)

    
    
    
def plot_histogram(self,separate_pores=False,do_legend=False,dt=1.0,start=0,end=None):
    age=arange(0,self.particle_age.max(),dt)
    if end is not None:
        end=int(end)
    start=int(start)
    bottom=zeros_like(age)[start:end]
    if separate_pores:
        
        # location_type=array(self.location_map[self.particle_location])[start:end]
        location_type=self.location_map[stack([self.get_hist_array(self.move_hist,n,dt,len(age))[1] for n in range(len(self.particle_location))])]
        forms=stack([self.get_hist_array(self.transform_hist,n,dt,len(age))[1] for n in range(len(self.particle_location))])
        old_bottom=bottom[-1]
        age=age[start:end]
        for poresize in (['macropore','micropore','nanopore']):
            for Ctype in reversed(['lignin','insoluble polymer','soluble polymer','monomer','microbe','CO2']):
                if Ctype is 'CO2':
                    label='CO$_2$'
                else:
                    label=Ctype.capitalize()
                top=bottom+((forms[:,start:end]==self.Ctypes.index(Ctype))&(location_type[:,start:end]==self.poretypes.index(poresize))).sum(axis=0)/forms.shape[0]
                fill_between(age,bottom,top,label=label,facecolor=self.plotcolors[Ctype],hatch=self.hatching.get(Ctype,None),edgecolor='k')
                bottom=top
            plot(age,top,'w:',lw=2.0)
            annotate(poresize[:-4].capitalize(),xytext=(age[-1]*1.03,(old_bottom+bottom[-1])*0.5),xy=(age[-1]*1.01,(old_bottom+bottom[-1])*0.5),
                        arrowprops={'arrowstyle':'-[, widthB=%1.1f, lengthB=0.3'%(abs(old_bottom-bottom[-1])*8.5)},va='center')
            # text(age[-1]*1.01,(old_bottom+bottom[-1])*0.5,poresize[:-4].capitalize(),rotation=0,va='bottom')
            old_bottom=bottom[-1]


        xlabel('Time (years)',fontsize='large')
        ylabel('Relative amount',fontsize='large')
        title('Relative amount of each particle type in each pore size')
        xlim(0,age[-1]*1.1)
        ylim(0,1)
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)

        
        
    else:
        
        for Ctype in ['microbe','lignin','insoluble polymer','soluble polymer','monomer','CO2']:
            top=bottom+(self.particle_form==self.Ctypes.index(Ctype)).sum(axis=0)[start:end]/self.particle_form.count(axis=0)
            fill_between(age,bottom,top,label=Ctype.capitalize(),color=self.plotcolors[Ctype])
            bottom=top

        if do_legend:
            legend(ncol=2,fontsize='small')
        xlabel('Time steps')
        ylabel('Relative amount')
        title('Relative total amount of each C type over time')




###########################################
###########   Run sims  ###################
###########################################


# Initial conditions


total_particles=0
np.random.seed(1)

# Initialize a simulation
nparticles=100
nlocations=nparticles*2
timestep=4/(24*365) # in years
sim_length=5 # years
ntimes=int(sim_length/timestep)

sim = lagrangian_soil_sim(nparticles=nparticles,nlocations=nlocations)
sim_immobile = lagrangian_soil_sim(nparticles=nparticles,nlocations=nlocations)
# sim_immobile.move_probs_matrix = sim_immobile.move_probs_matrix*2
sim_immobile.prob_leaving = sim_immobile.prob_leaving*0.1

sim_macro = lagrangian_soil_sim(nparticles=nparticles,nlocations=nlocations,pore_distribution=array([0.8,0.1,0.1]))
sim_macro_immobile = lagrangian_soil_sim(nparticles=nparticles,nlocations=nlocations,pore_distribution=array([0.8,0.1,0.1]))
# sim_immobile.move_probs_matrix = sim_immobile.move_probs_matrix*2
sim_macro_immobile.prob_leaving = sim_macro_immobile.prob_leaving*0.1

# Add some microbes
nmicrobes=nparticles//10
for ii in range(nmicrobes):
    sim.add_particle(ii,C_type='microbe')
    sim_immobile.add_particle(ii,C_type='microbe')
    sim_macro.add_particle(ii,C_type='microbe')
    sim_macro_immobile.add_particle(ii,C_type='microbe')
    total_particles += 1
for ii  in range(nparticles-nmicrobes):
    C_type=['lignin','insoluble polymer','soluble polymer'][randint(3)]
    sim.add_particle(total_particles,C_type=C_type,pore_type='random')
    sim_immobile.add_particle(total_particles,C_type=C_type,pore_type='random')
    sim_macro.add_particle(total_particles,C_type=C_type,pore_type='random')
    sim_macro_immobile.add_particle(total_particles,C_type=C_type,pore_type='random')
    total_particles += 1

# Iterate model
# Seed the pseudo-random number generator, for reproducible simulations
import time

for s in [sim,sim_immobile,sim_macro,sim_macro_immobile]:
    start=time.time()
    np.random.seed(1)
    for tt in range(1,ntimes):
        if tt%1000==0:
            print('Step %d of %d (%1.1f %%). Time elapsed: %1.1f s'%(tt,ntimes,tt/ntimes*100,time.time()-start))
            
        s.step()




###########################################
###########   Plots  ######################
###########################################

# Set time steps so total simulation length (10000) is 5 years
# This makes time step about 4 hours
sim.particle_age = sim.particle_age*timestep
sim_immobile.particle_age = sim_immobile.particle_age*timestep
sim_macro.particle_age = sim_macro.particle_age*timestep
sim_macro_immobile.particle_age = sim_macro_immobile.particle_age*timestep
# 
# historyfig=figure('Particle histories',figsize=(10,6))
# historyfig.clf()
# 
# subplot(221)
# sim.plot_particle_loc_history(dt=timestep,end=int(5/timestep))
# title('Low mobility, even pore distribution')
# xlabel('Time (years)')
# subplot(222)
# sim_immobile.plot_particle_loc_history(dt=timestep,end=int(5/timestep))
# title('High mobility, even pore distribution')
# xlabel('Time (years)')
# subplot(223)
# sim_macro.plot_particle_loc_history(dt=timestep,end=int(5/timestep))
# title('Low mobility, macropore-dominated')
# xlabel('Time (years)')
# subplot(224)
# sim_macro_immobile.plot_particle_loc_history(dt=timestep,end=int(5/timestep))
# title('High mobile, macropore-dominated')
# xlabel('Time (years)')
# 
# tight_layout()

# 
# cascadefig=figure('Particle tranformation cascades',figsize=(10,6))
# cascadefig.clf()
# subplot(221)
# plot_particle_cascade(sim,end=int(5/timestep))
# title('Low mobility, even pore distribution')
# xlabel('Time (years)')
# subplot(222)
# plot_particle_cascade(sim_immobile,end=int(5/timestep))
# title('High mobility, even pore distribution')
# xlabel('Time (years)')
# subplot(223)
# plot_particle_cascade(sim_macro,end=int(5/timestep))
# title('Low mobility, macropore-dominated')
# xlabel('Time (years)')
# subplot(224)
# plot_particle_cascade(sim_macro_immobile,end=int(5/timestep))
# title('High mobility, macropore-dominated')
# xlabel('Time (years)')



histogramfig=figure('Histograms',figsize=(12.5,6.8),clear=True)
axs=histogramfig.subplots(2,2)

sca(axs[0,1])
# sim_immobile.plot_histogram(separate_pores=False)
# subplot(245)
plot_histogram(sim_immobile,separate_pores=True,dt=0.01,end=5/.01)
# title('Divided by pore class')
title('Low mobility, even pore distribution')
xlabel('Time (years)')


sca(axs[0,0])
# sim.plot_histogram(separate_pores=False)
# subplot(246)
plot_histogram(sim,separate_pores=True,dt=0.01,end=5/.01)
# title('Divided by pore class')
title('High mobility, even pore distribution')
xlabel('Time (years)')


sca(axs[1,1])
# sim_macro_immobile.plot_histogram(separate_pores=False)
# subplot(247)
plot_histogram(sim_macro_immobile,separate_pores=True,dt=0.01,end=5/.01)
# title('Divided by pore class')
title('Low mobility, macropore-dominated')
xlabel('Time (years)')


sca(axs[1,0])
# sim_macro.plot_histogram(separate_pores=False)
# subplot(248)
plot_histogram(sim_macro,separate_pores=True,dt=0.01,end=5/.01)
title('High mobility, macropore-dominated')
xlabel('Time (years)')
# title('Divided by pore class')
leg=legend(handles=gca().collections[:6],fontsize='medium',loc=(0.25,0.035),framealpha=1.0,ncol=2)
leg.set_draggable(True)

# tight_layout()




figure('Particle history example');clf()
plot_particle_loc_history(sim,dt=0.01,do_legend=True,particles=[0,49,18,71,92,62],end=5/.01)
title('History of particle locations')
# legend(ncol=3)
# tight_layout()
xlabel('Time (years)')


# Bigger simulation for residence times
nparticles=400
nlocations=nparticles*2
timestep=4/(24*365) # in years
sim_length=200 # years
ntimes=int(sim_length/timestep)

sim_long = lagrangian_soil_sim(nparticles=nparticles,nlocations=nlocations)
# Add some microbes
nmicrobes=nparticles//10
total_particles=0
for ii in range(nmicrobes):
    sim_long.add_particle(ii,C_type='microbe')
    total_particles += 1
for ii  in range(nparticles-nmicrobes):
    C_type=['lignin','insoluble polymer','soluble polymer','monomer'][randint(4)]
    sim_long.add_particle(total_particles,C_type=C_type,pore_type='random')
    total_particles += 1

# Iterate model
# Seed the pseudo-random number generator, for reproducible simulations
import time

start=time.time()
np.random.seed(1)
for tt in range(1,ntimes):
    if tt%1000==0:
        print('Step %d of %d (%1.1f %%). Time elapsed: %1.1f s'%(tt,ntimes,tt/ntimes*100,time.time()-start))
    sim_long.step()
sim_long.particle_age=sim_long.particle_age*timestep

# Residence times in each chemical form (not taking pore location into account)
def res_times(sim,dt=timestep,poretype=None):
    times=zeros((len(sim.Ctypes),len(sim.particle_form)))
    for particle in range(sim.particle_form.shape[0]):
        form=sim.get_hist_array(sim.transform_hist,particle,tstep=dt)[1]
        loc=sim.get_hist_array(sim.move_hist,particle,tstep=dt)[1]
        if poretype is not None:
            times[:,particle]=histogram(form[sim.location_map[loc]==sim.poretypes.index(poretype)],bins=arange(0,len(sim.Ctypes)+1))[0]
        else:
            times[:,particle]=histogram(form,bins=arange(0,len(sim.Ctypes)+1))[0]
    return times*dt

times_all=res_times(sim_long,0.1) 
times_nano=res_times(sim_long,0.1,poretype='nanopore') 
times_micro=res_times(sim_long,0.1,poretype='micropore') 
times_macro=res_times(sim_long,0.1,poretype='macropore') 
t0_form=array([sim_long.get_hist_array(sim_long.transform_hist,num)[1][0] for num in range(len(sim_long.particle_form))])

f=figure('Residence times',clear=True)
bins=arange(0,sim_length,2.0)
# bins=concatenate(([0],logspace(0,2.5,10)  ))
gs=f.add_gridspec(nrows=2,ncols=1)
ax0=f.add_subplot(gs[0])
gs2=gs[1].subgridspec(ncols=4,nrows=1)
line_axs=[f.add_subplot(gs2[num]) for num in range(4)]
pnums=[]
pnames=[]
cm=get_cmap('gray_r')
for particletype in sim.Ctypes:
    if particletype in ['enzyme','CO2','microbe']:
        continue
    # ax0.plot((b[1:]+b[:-1])/2,h,label=particletype)
    x,y=meshgrid(b,[sim.Ctypes.index(particletype),sim.Ctypes.index(particletype)+0.2])
    t=times_all[sim.Ctypes.index(particletype),:]
    t_macro=times_macro[sim.Ctypes.index(particletype),:]
    t_micro=times_micro[sim.Ctypes.index(particletype),:]
    t_nano=times_nano[sim.Ctypes.index(particletype),:]
    h,b=histogram(t[t>0],bins,density=False)
    vmax=0.4
    m=ax0.pcolormesh(x,y,stack((h,h))/h.sum(),cmap=cm,vmin=0,vmax=vmax)
    ax0.plot(t[t>0].mean(),sim.Ctypes.index(particletype)+0.1,'wo',mec='k')
    h_macro,b=histogram(t_macro[t_macro>0],bins,density=False)
    ax0.pcolormesh(x,y+0.2,stack((h_macro,h_macro))/h.sum(),cmap=cm,vmin=0,vmax=vmax)
    ax0.plot(t_macro[t_macro>0].mean(),sim.Ctypes.index(particletype)+0.3,'wo',mec='k')
    h_micro,b=histogram(t_micro[t_micro>0],bins,density=False)
    ax0.pcolormesh(x,y+0.4,stack((h_micro,h_micro))/h.sum(),cmap=cm,vmin=0,vmax=vmax)
    ax0.plot(t_micro[t_micro>0].mean(),sim.Ctypes.index(particletype)+0.5,'wo',mec='k')
    h_nano,b=histogram(t_nano[t_nano>0],bins,density=False)
    ax0.pcolormesh(x,y+0.6,stack((h_nano,h_nano))/h.sum(),cmap=cm,vmin=0,vmax=vmax)
    ax0.plot(t_nano[t_nano>0].mean(),sim.Ctypes.index(particletype)+0.7,'wo',mec='k')
    
    ax0.axhline(y[0,0],ls=':',c='k',lw=0.5)
    ax0.axhline(y[0,0]+0.2,ls=':',c='k',lw=0.3)
    ax0.axhline(y[0,0]+0.8,ls=':',c='k',lw=0.5)
    
    ax0.text(x[0,-1]*0.99,y[0,0]+0.1,'All',ha='right',va='center')
    ax0.text(x[0,-1]*0.99,y[0,0]+0.1+0.2,'Macro',ha='right',va='center')
    ax0.text(x[0,-1]*0.99,y[0,0]+0.1+0.4,'Micro',ha='right',va='center')
    ax0.text(x[0,-1]*0.99,y[0,0]+0.1+0.6,'Nano',ha='right',va='center')

    h,b=histogram(t[t>0],bins,density=True)
    line_axs[0].plot(b,concatenate(([0],(diff(b)*h).cumsum())),label=particletype.capitalize(),ls='-',c='C'+str(sim.Ctypes.index(particletype)))
    h,b=histogram(t_macro[t_macro>0],bins,density=True)
    line_axs[1].plot(b,concatenate(([0],(diff(b)*h).cumsum())),ls='-',c='C'+str(sim.Ctypes.index(particletype)))
    h,b=histogram(t_micro[t_micro>0],bins,density=True)
    line_axs[2].plot(b,concatenate(([0],(diff(b)*h).cumsum())),ls='-',c='C'+str(sim.Ctypes.index(particletype)))
    h,b=histogram(t_nano[t_nano>0],bins,density=True)
    line_axs[3].plot(b,concatenate(([0],(diff(b)*h).cumsum())),ls='-',c='C'+str(sim.Ctypes.index(particletype)))
    
    pnums.append(sim.Ctypes.index(particletype))
    pnames.append(particletype.capitalize())

for num in range(4):
    line_axs[num].set_xlabel('Residence time (years)')
    line_axs[num].set_ylabel('Cumulative fraction')
line_axs[0].legend()
line_axs[0].set_title('All pore sizes')
line_axs[1].set_title('Macropores')
line_axs[2].set_title('Micropores')
line_axs[3].set_title('Nanopores')
ax0.set_yticks(array(pnums)+0.4)
ax0.set_yticks(concatenate((array(pnums),array(pnums)+0.8)),True)
ax0.set_yticklabels(pnames)
ax0.tick_params(length=0,which='major')
ax0.set_xlabel('Residence time (years)')
cb=colorbar(m,ax=ax0)
cb.set_label('Fraction of particles')

figure('Big sim',clear=True)
plot_histogram(sim_long,separate_pores=True,dt=0.1)
legend(handles=gca().collections[:6],fontsize='medium',loc=(0.25,0.035),framealpha=1.0,ncol=2)

show()
