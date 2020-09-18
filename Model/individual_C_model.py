from pylab import *


class lagrangian_soil_sim:
    
    def __init__(self,pore_distribution=array([0.3,0.4,0.3]),nlocations=200,nparticles=100,dt=4/(24*365)):
        self.Ctypes=['CO2','microbe','monomer','soluble polymer','insoluble polymer','lignin','enzyme']
        # self.plotcolors={'lignin':'C0','insoluble polymer':'C1','soluble polymer':'C2','monomer':'C3','microbe':'C4','CO2':'C5','enzyme':'C6'}
        self.plotcolors={'lignin':[0,0,0],'insoluble polymer':[0.4,0.4,0.4],'soluble polymer':[0.7,0.7,0.7],'monomer':[0.9,0.9,0.9],'microbe':[1.0,1.0,1.0],'CO2':[1.0,1.0,1.0],'enzyme':'C6'}
        self.hatching={'microbe':'||'}
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
        self.transform_matrix[self.Ctypes.index('lignin'),self.Ctypes.index('soluble polymer')]=.25e-2
        self.transform_matrix[self.Ctypes.index('insoluble polymer'),self.Ctypes.index('soluble polymer')]=1e-2
        self.transform_matrix[self.Ctypes.index('insoluble polymer'),self.Ctypes.index('monomer')]=1e-2
        self.transform_matrix[self.Ctypes.index('soluble polymer'),self.Ctypes.index('monomer')]=2e-2
        self.transform_matrix[self.Ctypes.index('monomer'),self.Ctypes.index('CO2')]=1e-1
        
        self.prob_leaving=array([0.05,0.005,0.0005])
            
        self.move_probs_matrix=zeros((len(self.Ctypes),len(self.poretypes)))
        self.move_probs_matrix[self.Ctypes.index('lignin'),:]=array([0.01,0.01,0.0])
        self.move_probs_matrix[self.Ctypes.index('insoluble polymer'),:]=array([0.01,0.01,0.0])
        self.move_probs_matrix[self.Ctypes.index('soluble polymer'),:]=array([0.1,0.05,0.005])
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
        out[int(hist_p[-1]['time']*self.dt/tstep):]=hist_p[-1]['to']
        
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

new_pore_names={'macropore':'Flow-permitting','micropore':'Interparticle','nanopore':'Intraparticle'}
porecolors={'macropore':'g','micropore':'b','nanopore':'orange'}

def plot_particle_loc_history(self,do_legend=False,dt=1,particles=None,draw_pores=[],start=0,end=None):
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
        offset=-len(self.location_map)/40*0.5
        for num in range(len(bounds)-1):
            Ctype=self.Ctypes[form[bounds[num]]]
            if Ctype=='CO2':
                continue
            r=Rectangle(xy=(age[bounds[num]],location[bounds[num]]+offset),width=age[bounds[num+1]]-age[bounds[num]],height=len(self.location_map)/40,
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
                # scatter(age[max(0,trans-1)],location[max(0,trans-1)],marker='x',s=30.0,facecolor=self.plotcolors[self.Ctypes[form[max(0,trans-1)]]],zorder=100)
                scatter(age[max(0,trans-1)],location[max(0,trans-1)],marker='*',s=60.0,facecolor='y',edgecolor='r',zorder=100,alpha=0.5)
            else:
                # scatter(age[max(0,trans-1)],location[max(0,trans-1)],marker='o',s=20.0,edgecolor=self.plotcolors[self.Ctypes[form[max(0,trans-1)]]],facecolor='None',zorder=100)
                scatter(age[max(0,trans-1)],location[max(0,trans-1)],marker='o',s=40.0,edgecolor='r',facecolor='y',zorder=100,alpha=0.5)

        for mov in movements:
            if mov>=len(location):
                continue
            annotate('',xytext=(age[max(0,mov-1)],location[max(0,mov-1)]),xy=(age[mov],location[mov]),color=self.plotcolors[self.Ctypes[form[mov]]],
                        arrowprops={'arrowstyle':'->'})
    
    boundaries=nonzero(diff(self.location_map)!=0)[0]
    poresizes=['Macropore','Micropore','Nanopore']
    # text(age[-1],0+2.5,poresizes[self.location_map[0]],rotation=60,va='bottom',fontsize='small')
    b=concatenate(([0],boundaries,[len(self.location_map)-1]))
    for num in range(len(b)-1):
        annotate(new_pore_names[poresizes[num].lower()],xytext=(age[-1]*1.03,(b[num]+b[num+1])*0.5),xy=(age[-1]*1.02,(b[num]+b[num+1])*0.5),
                    arrowprops={'arrowstyle':'-[, widthB=%1.1f, lengthB=0.4'%(abs(b[num]-b[num+1])*1.9e-2)},va='center')
        gca().add_patch(Rectangle((age[0],b[num]),width=age[-1],height=b[num+1]-b[num],facecolor=porecolors[poresizes[num].lower()],alpha=0.2))
    p=Polygon(array([[age[-1]*1.2,0],[age[-1]*1.3,0],[age[-1]*1.25,len(self.location_map)]]),facecolor=[0.25,0.25,0.25])
    gca().add_patch(p)
    text(age[-1]*1.25,len(self.location_map)/2,'Pore size class',color='w',ha='center',va='center',rotation=90)
    for num in boundaries:
        plot(age,zeros(len(age))+num+0.5,'k--',lw=2)
        # text(age[-1],num+2.5,poresizes[self.location_map[num+1]],rotation=60,va='bottom',fontsize='small')
    for num in draw_pores:
        plot([age[0],age[-1]],zeros(2)+num,'k--',lw=0.2,color=[0.5,0.5,0.5])

    Ctypes=['lignin','insoluble polymer','soluble polymer','monomer','microbe']
    if do_legend:
        legend([Rectangle(xy=(0,0),width=0,height=2.5,
            facecolor=self.plotcolors[Ctype],hatch=self.hatching.get(Ctype,None),edgecolor='k') for Ctype in Ctypes]+\
                [Line2D([0],[0],marker='o',ls='None',markeredgecolor='r',markerfacecolor='y',ms=10.0),
                Line2D([0],[0],marker='*',ls='None',markeredgecolor='r',markerfacecolor='r',ms=10.0)],
            [c.capitalize() for c in Ctypes]+['C transformation','CO$_2$ produced'],loc=(0.0,1.08),ncol=3)

    xlabel('Time (years)',fontsize='large')
    ylabel('Individual pore spaces',fontsize='large')
    yticks(draw_pores,labels=[])
    title('History of particle location')
    ylim(0,len(self.location_map)+1)
    gca().spines['right'].set_visible(False)
    xlim(0,age[-1]*1.3)




def plot_histogram(self,separate_pores=False,do_legend=False,dt=1.0,start=0,end=None,showCO2=True,doplot=True):
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
        if not doplot:
            return (age,forms,location_type)
        for poresize in (['macropore','micropore','nanopore']):
            pore_bottom=bottom
            for Ctype in reversed(['lignin','insoluble polymer','soluble polymer','monomer','microbe','CO2']):
                if Ctype is 'CO2':
                    label='CO$_2$'
                    if not showCO2:
                        continue
                else:
                    label=Ctype.capitalize()
                if showCO2:
                    top=bottom+((forms[:,start:end]==self.Ctypes.index(Ctype))&(location_type[:,start:end]==self.poretypes.index(poresize))).sum(axis=0)/forms.shape[0]
                else:
                    top=bottom+((forms[:,start:end]==self.Ctypes.index(Ctype))&(location_type[:,start:end]==self.poretypes.index(poresize))).sum(axis=0)/(forms!=self.Ctypes.index('CO2')).sum(axis=0)
                fill_between(age,bottom,top,label=label,facecolor=self.plotcolors[Ctype],hatch=self.hatching.get(Ctype,None),edgecolor='k')
                bottom=top
                if showCO2 and Ctype is 'CO2':
                    pore_bottom=bottom
            plot(age,top,'-',lw=2.0,c=porecolors[poresize])
            fill_between(age,pore_bottom,top,facecolor=porecolors[poresize],alpha=0.15,zorder=4)
            annotate(new_pore_names[poresize],xytext=(age[-1]*1.03,(old_bottom+bottom[-1])*0.5),xy=(age[-1]*1.01,(old_bottom+bottom[-1])*0.5),
                        arrowprops={'arrowstyle':'-[, widthB=%1.1f, lengthB=0.3'%(abs(old_bottom-bottom[-1])*8.5)},va='center',color=porecolors[poresize])
            # text(age[-1]*1.01,(old_bottom+bottom[-1])*0.5,poresize[:-4].capitalize(),rotation=0,va='bottom')
            old_bottom=bottom[-1]


        xlabel('Time (years)',fontsize='large')
        ylabel('Relative amount',fontsize='large')
        title('Relative amount of each particle type in each pore size')
        xlim(0,age[-1]*1.1)
        ylim(0,1)
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        
        return (age,forms)

        
        
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
nparticles=400
nlocations=nparticles*2
timestep=4/(24*365) # in years
sim_length=5 # years
ntimes=int(sim_length/timestep)

sim = lagrangian_soil_sim(nparticles=nparticles,nlocations=nlocations)
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
    C_type=['lignin','insoluble polymer','soluble polymer','monomer'][randint(4)]
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
axs=histogramfig.subplots(2,2,sharex=False,sharey=False)

sca(axs[0,1])
# sim_immobile.plot_histogram(separate_pores=False)
# subplot(245)
plot_histogram(sim_immobile,separate_pores=True,dt=0.01,end=5/.01)
# title('Divided by pore class')
# title('Low mobility, even pore distribution')
# xlabel('Time (years)')
text(0.02,1.02,'(b)',fontsize='medium')
text(0.5,1.05,'Low soil moisture',fontsize='x-large',transform=axs[0,1].transAxes,ha='center')
title('')
ylabel('')
xlabel('')


sca(axs[0,0])
# sim.plot_histogram(separate_pores=False)
# subplot(246)
plot_histogram(sim,separate_pores=True,dt=0.01,end=5/.01)
# title('Divided by pore class')
# title('High mobility, even pore distribution')
# xlabel('Time (years)')
text(0.02,1.02,'(a)',fontsize='medium')
text(0.5,1.05,'High soil moisture',fontsize='x-large',transform=axs[0,0].transAxes,ha='center')
text(-0.15,0.5,'Fine textured soil',fontsize='x-large',transform=axs[0,0].transAxes,va='center',rotation=90)
title('')
xlabel('')


sca(axs[1,1])
# sim_macro_immobile.plot_histogram(separate_pores=False)
# subplot(247)
plot_histogram(sim_macro_immobile,separate_pores=True,dt=0.01,end=5/.01)
# title('Divided by pore class')
# title('Low mobility, macropore-dominated')
xlabel('Time (years)')
text(0.02,1.02,'(d)',fontsize='medium')
title('')
ylabel('')


sca(axs[1,0])
# sim_macro.plot_histogram(separate_pores=False)
# subplot(248)
plot_histogram(sim_macro,separate_pores=True,dt=0.01,end=5/.01)
# title('High mobility, macropore-dominated')
xlabel('Time (years)')
# title('Divided by pore class')
leg=legend(handles=gca().collections[:6],fontsize='medium',loc=(0.48,0.02),framealpha=1.0,ncol=2)
leg.set_draggable(True)
text(0.02,1.02,'(c)',fontsize='medium')
text(-0.15,0.5,'Coarse textured soil',fontsize='x-large',transform=axs[1,0].transAxes,va='center',rotation=90)
title('')

# tight_layout()

# Moved at least once:
moved=[]
moved_trans=[]
stuck_nano=[]

for part in range(len(sim.particle_location)):
    transforms=sim.get_hist_array(sim.transform_hist,part,tstep=0.1)[0] 
    movements=sim.get_hist_array(sim.move_hist,part,tstep=0.1)[0] 
    if len(movements)>1:
        moved.append(part)
    if len(movements)>1 and len(transforms) > 1:
        moved_trans.append(part)
    if len(movements)>1 and sim.location_map[sim.particle_location[part]]==2 and sim.particle_form[part] != 0:
        stuck_nano.append(part)
        
def get_microbes(part,dt=0.01):
    t,form_hist=sim.get_hist_array(sim.transform_hist,part,tstep=dt)
    m,loc_hist=sim.get_hist_array(sim.move_hist,part,tstep=dt)
    microbes=[]
    microbe_locs=stack([sim.get_hist_array(sim.move_hist,part,tstep=dt)[1] for part in range(nmicrobes)])
    for n,tt in enumerate(t[1:]):
        timepoint=int(tt['time']*sim.dt/dt)
        loc=loc_hist[timepoint]
        mloc=nonzero(microbe_locs[:,timepoint]==loc)[0]
        if len(mloc)>0:
            microbes.append(mloc)
        elif len(nonzero(microbe_locs[:,timepoint]==loc_hist[timepoint-1])[0])>0:
            microbes.append(nonzero(microbe_locs[:,timepoint]==loc_hist[timepoint-1])[0])
        elif len(nonzero(microbe_locs[:,timepoint]==loc_hist[timepoint+1])[0])>0:
            microbes.append(nonzero(microbe_locs[:,timepoint]==loc_hist[timepoint+1])[0])
        else:
            # raise ValueError('Microbe not found')
            print('Warning: Microbe not found for particle %d'%part)
            return []
    return concatenate(microbes)
    


figure('Particle history example');clf()
particles=[0,49,18,71,92,62] # Previous plot, but sims changed
particles=[]
particles.append(stuck_nano[0])
particles.append(moved_trans[1])
m=get_microbes(moved_trans[1])[0]
particles.append(m)
for part in moved_trans[1:]:
    if m in get_microbes(part):
        particles.append(part)
# particles.remove(200)
plot_particle_loc_history(sim,dt=0.01,do_legend=True,particles=particles[1:7],end=4/.01,draw_pores=linspace(0,len(sim.location_map),40))
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
age,form,loc=plot_histogram(sim_long,separate_pores=True,dt=0.1,doplot=False) 

f=figure('Residence times',clear=True)
bins=arange(0,sim_length,2.0)
bins=arange(0,151,2.0)
# bins=concatenate(([0],logspace(0,2.5,10)  ))
gs=f.add_gridspec(nrows=1,ncols=2,width_ratios=[2,10])
ax0=f.add_subplot(gs[1])
gs2=gs[0].subgridspec(ncols=1,nrows=6,height_ratios=[.05,1,1,1,1,0.1])
pie_axs=[f.add_subplot(gs2[num+1]) for num in reversed(range(4))]
# ax0=f.subplots(1,1)
pnums=[]
pnames=[]
cm_all=get_cmap('gray_r')
cm_macro=get_cmap('Greens')
cm_micro=get_cmap('Blues')
cm_nano=get_cmap('Oranges')
medianprops={'color':'k'}
meanprops={'marker':'o','linestyle':'None','markerfacecolor':'None','markeredgecolor':'k','markersize':5}
flierprops={'marker':'o','markeredgewidth':0.5,'markersize':2}
boxprops={'linewidth':2.0}
whiskerprops={}

def distplot(x,y,data,norm_by=None,color=None,cmap=cm_all,vmax=0.4,widths=0.07,show_hist=False,
    flierprops=flierprops,meanprops=meanprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops,**kwargs):
    if color is None:
        color=cmap(1.0)
        
    if show_hist:
        h,b=histogram(data,bins,density=False)
        if norm_by is None:
            norm_by=h.sum()
        m=ax0.pcolormesh(x,y,stack((h,h))/norm_by,cmap=cmap,vmin=0,vmax=vmax)
    else:
        m=None
    
    flierprops=flierprops.copy()
    flierprops.update(color=color,markeredgecolor=color,markerfacecolor=color)
    medianprops=medianprops.copy()
    medianprops.update(color=color)
    meanprops=meanprops.copy()
    meanprops.update(markeredgecolor=color)
    boxprops=boxprops.copy()
    boxprops.update(color=color)
    whiskerprops=whiskerprops.copy()
    whiskerprops.update(color=color)
    
    ax0.boxplot(data,vert=False,positions=[y[0,0]+0.1],showmeans=True,whis=[5,95],showfliers=False,
        flierprops=flierprops,medianprops=medianprops,meanprops=meanprops,boxprops=boxprops,whiskerprops=whiskerprops,widths=widths,showcaps=False,**kwargs)
        
    return m


for particletype in sim.Ctypes:
    if particletype in ['enzyme','CO2','microbe']:
        continue
    # ax0.plot((b[1:]+b[:-1])/2,h,label=particletype)
    x,y=meshgrid(bins,[sim.Ctypes.index(particletype),sim.Ctypes.index(particletype)+0.2])
    t=times_all[sim.Ctypes.index(particletype),:]
    t_macro=times_macro[sim.Ctypes.index(particletype),:]
    t_micro=times_micro[sim.Ctypes.index(particletype),:]
    t_nano=times_nano[sim.Ctypes.index(particletype),:]
    m=distplot(x,y,t[t>0],color='k')
    distplot(x,y+0.2,t_macro[t_macro>0],cmap=cm_all,color=porecolors['macropore'])
    distplot(x,y+0.4,t_micro[t_micro>0],cmap=cm_all,color=porecolors['micropore'])
    distplot(x,y+0.6,t_nano[t_nano>0],cmap=cm_all,color=porecolors['nanopore'])
    
    # ax0.axhspan(ymin=sim.Ctypes.index(particletype),ymax=sim.Ctypes.index(particletype)+0.8,color=sim_long.plotcolors[particletype])
    ax0.axhspan(ymin=sim.Ctypes.index(particletype)+0.2,ymax=sim.Ctypes.index(particletype)+0.4,color=porecolors['macropore'],alpha=0.15)
    ax0.axhspan(ymin=sim.Ctypes.index(particletype)+0.4,ymax=sim.Ctypes.index(particletype)+0.6,color=porecolors['micropore'],alpha=0.15)
    ax0.axhspan(ymin=sim.Ctypes.index(particletype)+0.6,ymax=sim.Ctypes.index(particletype)+0.8,color=porecolors['nanopore'],alpha=0.15)
    
    maxage=50
    num_macro=((age<maxage)&(form==sim.Ctypes.index(particletype))&(loc==sim.poretypes.index('macropore'))).sum()
    num_micro=((age<maxage)&(form==sim.Ctypes.index(particletype))&(loc==sim.poretypes.index('micropore'))).sum()
    num_nano=((age<maxage)&(form==sim.Ctypes.index(particletype))&(loc==sim.poretypes.index('nanopore'))).sum()
    pie_axs[sim.Ctypes.index(particletype)-2].pie([num_nano,num_micro,num_macro],colors=[porecolors['nanopore'],porecolors['micropore'],porecolors['macropore']],wedgeprops={'linewidth':1.5,'joinstyle':'bevel'})
    for w in pie_axs[sim.Ctypes.index(particletype)-2].patches:
        if w.theta1 != w.theta2:
            c=w.get_facecolor()
            w.set_edgecolor(c)
            w.set_facecolor(c[:3]+(0.15,))
    pie_axs[sim.Ctypes.index(particletype)-2].text(0,0.5,particletype.replace(' ','\n').capitalize()+'\ndistribution',fontsize='medium',ha='right',va='center',transform=pie_axs[sim.Ctypes.index(particletype)-2].transAxes)
    pie_axs[sim.Ctypes.index(particletype)-2].set_ylim(-1.25,1.05)
    # pie_axs[sim.Ctypes.index(particletype)-2].add_patch(Circle((0,0),1.0,facecolor=sim_long.plotcolors[particletype],zorder=-1))
    # pie_axs[sim.Ctypes.index(particletype)-2].set_title(particletype.capitalize())
    
    # h_macro,b=histogram(t_macro[t_macro>0],bins,density=False)
    # ax0.pcolormesh(x,y+0.2,stack((h_macro,h_macro))/h.sum(),cmap=cm_macro,vmin=0,vmax=vmax)
    # ax0.boxplot(t_macro[t_macro>0],vert=False,positions=[sim.Ctypes.index(particletype)+0.3],widths=0.1,
    #     flierprops=flierprops,medianprops=medianprops.update(color='g'),showmeans=True,meanprops=meanprops,boxprops={'color':'g'},whiskerprops={'color':'g'})
    # h_micro,b=histogram(t_micro[t_micro>0],bins,density=False)
    # ax0.pcolormesh(x,y+0.4,stack((h_micro,h_micro))/h.sum(),cmap=cm_micro,vmin=0,vmax=vmax)
    # ax0.boxplot(t_micro[t_micro>0],vert=False,positions=[sim.Ctypes.index(particletype)+0.5],widths=0.1,flierprops=flierprops,meanprops=meanprops,medianprops=medianprops,showmeans=True,boxprops={'color':'b'})
    # h_nano,b=histogram(t_nano[t_nano>0],bins,density=False)
    # ax0.pcolormesh(x,y+0.6,stack((h_nano,h_nano))/h.sum(),cmap=cm_nano,vmin=0,vmax=vmax)
    # ax0.boxplot(t_nano[t_nano>0],vert=False,positions=[sim.Ctypes.index(particletype)+0.7],widths=0.1,flierprops=flierprops,meanprops=meanprops,medianprops=medianprops,showmeans=True,boxprops={'color':'orange'})
    # 
    ax0.axhline(y[0,0],ls=':',c='k',lw=0.5)
    ax0.axhline(y[0,0]+0.2,ls=':',c='k',lw=0.3)
    ax0.axhline(y[0,0]+0.8,ls=':',c='k',lw=0.5)
    
    ax0.text(175,y[0,0]+0.1,'All',ha='left',va='center')
    ax0.text(175,y[0,0]+0.1+0.2,'Flow-permitting',ha='left',va='center',color=porecolors['macropore'])
    ax0.text(175,y[0,0]+0.1+0.4,'Interparticle',ha='left',va='center',color=porecolors['micropore'])
    ax0.text(175,y[0,0]+0.1+0.6,'Intraparticle',ha='left',va='center',color=porecolors['nanopore'])
    # 
    # h,b=histogram(t[t>0],bins,density=True)
    # line_axs[0].plot(b,concatenate(([0],(diff(b)*h).cumsum())),label=particletype.capitalize(),ls='-',c='C'+str(sim.Ctypes.index(particletype)))
    # h,b=histogram(t_macro[t_macro>0],bins,density=True)
    # line_axs[1].plot(b,concatenate(([0],(diff(b)*h).cumsum())),ls='-',c='C'+str(sim.Ctypes.index(particletype)))
    # h,b=histogram(t_micro[t_micro>0],bins,density=True)
    # line_axs[2].plot(b,concatenate(([0],(diff(b)*h).cumsum())),ls='-',c='C'+str(sim.Ctypes.index(particletype)))
    # h,b=histogram(t_nano[t_nano>0],bins,density=True)
    # line_axs[3].plot(b,concatenate(([0],(diff(b)*h).cumsum())),ls='-',c='C'+str(sim.Ctypes.index(particletype)))
    
    pnums.append(sim.Ctypes.index(particletype))
    # pnames.append(particletype.capitalize().replace(' ','\n'))
    pnames.append('')


# line_axs[0].legend()
# line_axs[0].set_title('All pore sizes')
# line_axs[1].set_title('Macropores')
# line_axs[2].set_title('Micropores')
# line_axs[3].set_title('Nanopores')
ax0.set_yticks(array(pnums)+0.4)
ax0.set_yticks(concatenate((array(pnums),array(pnums)+0.8)),True)
ax0.set_yticklabels(pnames)
ax0.tick_params(length=0,which='major')
ax0.set_xlabel('Residence time (years)')
ax0.set_xlim(0,170)
ax0.set_ylim(min(pnums),max(pnums)+0.8) 
ax0.set_title('Distribution of residence times')
# pie_axs[-1].set_title('Distribution among pores')
# cb=colorbar(m,ax=ax0,aspect=30)
# cb.set_label('Fraction of particles')



figure('Big sim',clear=True)
plot_histogram(sim_long,separate_pores=True,dt=0.1,showCO2=False)
legend(handles=gca().collections[:6],fontsize='medium',loc=(0.25,0.035),framealpha=1.0,ncol=2)

show()
