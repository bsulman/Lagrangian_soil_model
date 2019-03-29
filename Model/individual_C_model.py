from pylab import *

Ctype_key={'enzyme':6,'lignin':5,'insoluble polymer':4,'soluble polymer':3,'monomer':2,'microbe':1,'CO2':0}
Ctype_key_inv=Ctype_key.__class__(map(reversed,Ctype_key.items()))
colors={'lignin':'C0','insoluble polymer':'C1','soluble polymer':'C2','monomer':'C3','microbe':'C4','CO2':'C5','enzyme':'C6'}

# pore_distribution={'macropore':0.3,'micropore':0.4,'nanopore':0.3}
pore_distribution=array([0.3,0.4,0.3])
pore_key={'macropore':0,'micropore':1,'nanopore':2}

nlocations=200

location_map=zeros(nlocations,dtype=int)


location_map[0:int(pore_distribution[0]*nlocations)]=0
location_map[int(pore_distribution[0]*nlocations):int((pore_distribution[1]+pore_distribution[0])*nlocations)]=pore_key['micropore']
location_map[int((pore_distribution[1]+pore_distribution[0])*nlocations):]=pore_key['nanopore']


nparticles=100
ntimes=5000
particle_age=ma.masked_all((nparticles,ntimes),dtype=int)
particle_form=ma.masked_all((nparticles,ntimes),dtype=int)
particle_location=ma.masked_all((nparticles,ntimes),dtype=int)

def add_particle(pnumber,time,pore_type='macropore',C_type='insoluble polymer'):
    locs=nonzero(location_map==pore_key[pore_type])[0]
    particle_location[pnumber,time]=locs[randint(len(locs))]
    particle_age[pnumber,time]=0
    particle_form[pnumber,time]=Ctype_key[C_type]


transform_matrix=zeros((len(Ctype_key),len(Ctype_key)))
transform_matrix[Ctype_key['lignin'],Ctype_key['soluble polymer']]=1e-2
transform_matrix[Ctype_key['insoluble polymer'],Ctype_key['soluble polymer']]=1e-2
transform_matrix[Ctype_key['insoluble polymer'],Ctype_key['monomer']]=1e-2
transform_matrix[Ctype_key['soluble polymer'],Ctype_key['monomer']]=2e-2
transform_matrix[Ctype_key['monomer'],Ctype_key['CO2']]=1e-1

def transform_particle(pnumber,time):
    current_type=particle_form[pnumber,time]
    current_location=particle_location[pnumber,time]
    others_in_pore=(particle_location[:,time]==current_location)

    microbes=sum(particle_form[others_in_pore,time]==Ctype_key['microbe'])
    enzymes=sum(particle_form[others_in_pore,time]==Ctype_key['enzyme'])
    transform_probs = transform_matrix[current_type,:]*(microbes+enzymes)
    # if current_type == Ctype_key['lignin']:
    #     prob['soluble polymer']=1e-2*microbes+1e-2*enzymes
    # if current_type == Ctype_key['insoluble polymer']:
    #     prob['soluble polymer']=1e-2*microbes+1e-2*enzymes
    #     prob['monomer']=1e-2*microbes+1e-2*enzymes
    # if current_type == Ctype_key['soluble polymer']:
    #     prob['monomer']=2e-2*microbes+2e-2*enzymes
    # if  current_type == Ctype_key['monomer']:
    #     prob['CO2']=1e-1*microbes
 
    
    prob = transform_probs-rand(len(transform_probs))
    if prob.max()>0:
        newtype = prob.argmax()
        print('Transformation! {old:s} --> {new:s}, prob={prob:1.2f}, pore={pore:d}'.format(old=Ctype_key_inv[current_type],new=Ctype_key_inv[newtype],prob=transform_probs[newtype],pore=current_location))
        return newtype
        
    # nprobs=len(prob.keys())
    # if nprobs>=1:
    #     for key in prob.keys():
    #         x=rand()
    #         if x<prob[key]:
    #             print('Transformation! {old:s} --> {new:s}, prob={prob:1.2f}, pore={pore:d}'.format(old=Ctype_key_inv[current_type],new=key,prob=prob[key],pore=current_location))
    #             return Ctype_key[key]

    return current_type


def transform_particles(time):
    microbe_pores = particle_location[particle_form[:,time]==Ctype_key['microbe'],time]
    enzyme_pores  = particle_location[particle_form[:,time]==Ctype_key['enzyme'],time] 
    
    transform_probs = zeros((len(particle_location),len(Ctype_key)))
    for micpore in microbe_pores:
        transform_probs[particle_location[:,time]==micpore,:]+=transform_matrix[particle_form[particle_location[:,time]==micpore,time],:]
    for enzpore in enzyme_pores:
        transform_probs[particle_location[:,time]==enzpore,:]+=transform_matrix[particle_form[particle_location[:,time]==enzpore,time],:]
    
    diceroll = transform_probs - random_sample(transform_probs.shape)
    transformed = diceroll.max(axis=1)>0
    transformed_to_type = diceroll.argmax(axis=1)
    
    for transnum in nonzero(transformed)[0]:
        print('Transformation! {old:s} --> {new:s}, prob={prob:1.2f}, pore={pore:d}'.format(old=Ctype_key_inv[particle_form[transnum,time]],
                                new=Ctype_key_inv[transformed_to_type[transnum]],prob=transform_probs[transnum,transformed_to_type[transnum]],pore=particle_location[transnum,time]))
    
    newtype = particle_form[:,time]
    newtype[transformed]=transformed_to_type[transformed]

    return newtype

prob_leaving=array([0.1,0.05,0.005])

move_probs = {
    'lignin':array([0.01,0.01,0.001]),
    'insoluble polymer':array([0.01,0.01,0.001]),
    'soluble polymer':array([0.1,0.05,0.05]),
    'monomer':array([0.1,0.1,0.1]),
    'microbe':array([0.1,0.01,0.0]),
    'enzyme':array([0.1,0.05,0.05]),
    }
    
move_probs_matrix=zeros((len(Ctype_key),len(pore_key)))
move_probs_matrix[Ctype_key['lignin'],:]=array([0.01,0.01,0.001])
move_probs_matrix[Ctype_key['insoluble polymer'],:]=array([0.01,0.01,0.001])
move_probs_matrix[Ctype_key['soluble polymer'],:]=array([0.1,0.05,0.05])
move_probs_matrix[Ctype_key['monomer'],:]=array([0.1,0.1,0.1])
move_probs_matrix[Ctype_key['microbe'],:]=array([0.1,0.01,0.0])
move_probs_matrix[Ctype_key['enzyme'],:]=array([0.1,0.05,0.05])

def move_particle(pnumber,time):
    current_location=particle_location[pnumber,time]
    current_location_type=location_map[current_location]
    current_type=particle_form[pnumber,time]

    x=rand(len(prob_leaving))
    prob=(prob_leaving[current_location_type]*move_probs[Ctype_key_inv[current_type]]*pore_distribution - x)
    if prob.max()>0:
        newporetype = prob.argmax()
        locs=nonzero(location_map==newporetype)[0]
        new_location=locs[randint(len(locs))]
        return new_location

    # for key in prob.keys():
    #     x=rand()
    #     # print(x,prob[key]*prob_leaving[current_location_type]*pore_distribution[key])
    #     if x<prob[key]*prob_leaving[current_location_type]*pore_distribution[key]:
    #         locs=nonzero(location_map==pore_key[key])[0]
    #         new_location=locs[randint(len(locs))]
    #         # print('Movement!!',current_location,new_location)
    #         return new_location

    return current_location

def move_particles(time):
    # Calculate movement probabilities for all particles
    moveprobs=move_probs_matrix[particle_form[:,time]]*prob_leaving[location_map[particle_location[:,time]],None]*pore_distribution[location_map[particle_location[:,time]],None]
    diceroll = moveprobs - random_sample(moveprobs.shape)
    moved = diceroll.max(axis=1)>0
    moved_to_type = diceroll.argmax(axis=1)
    destination = particle_location[:,time]
    for movenum in nonzero(moved)[0]:
        locs=nonzero(location_map==moved_to_type[movenum])[0]
        destination[movenum]=locs[randint(len(locs))]
    return destination

# Initial conditions
# Seed the pseudo-random number generator, for reproducible simulations
np.random.seed(1)

total_particles=0

# Add some microbes
nmicrobes=5
for ii in range(nmicrobes):
    add_particle(ii,0,C_type='microbe')
    total_particles += 1
for ii  in range(nparticles-nmicrobes):
    add_particle(total_particles,0,C_type=['lignin','insoluble polymer','soluble polymer'][randint(3)])
    total_particles += 1

# Iterate model
for tt in range(1,ntimes):
    if tt%100==0:
        print(tt)
    # for pnumber in range(total_particles):
        # if particle_form[pnumber,tt-1] == Ctype_key['CO2']:
        #     particle_form[pnumber,tt]=particle_form[pnumber,tt-1]
        #     particle_location[pnumber,tt]=particle_location[pnumber,tt-1]
        #     particle_age[pnumber,tt]=particle_age[pnumber,tt-1]+1
        #     continue
        # particle_form[pnumber,tt]=transform_particle(pnumber,tt-1)
        
    isCO2=particle_form[:,tt-1]==Ctype_key['CO2']
    particle_location[:,tt]=move_particles(tt-1)
    particle_age[~isCO2,tt]=particle_age[~isCO2,tt-1]+1
    particle_age[isCO2,tt]=particle_age[isCO2,tt-1]
    particle_form[:,tt]=transform_particles(tt-1)



###########################################
###########   Plots  ######################
###########################################

histfig=figure('Particle histories',figsize=(13,6))
histfig.clf()

subplot(121)
for pnum in range(total_particles):
    form=particle_form[pnum,:]
    location=particle_location[pnum,:]
    location_type=location_map[location]
    age=particle_age[pnum,:]

    pore_lines=['-','--',':']
    offset=rand()*0.2
    c='C%d'%(pnum%10)
    for loctype in pore_key.values():
        plot(age,ma.masked_array(form,mask=~(location_type==loctype)|(form==Ctype_key['microbe']))+offset,ls=pore_lines[loctype],c=c)

yticks(list(Ctype_key.values()),list(Ctype_key.keys()))
ylim(0.2,5.5)

legend([Line2D([0],[0],ls='-'),Line2D([0],[0],ls='--'),Line2D([0],[0],ls=':')],['macropore','micropore','nanopore'])

xlabel('Particle age')
title('History of particle C type')

subplot(122)
for pnum in range(total_particles):
    form=particle_form[pnum,:]
    location=particle_location[pnum,:]
    location_type=location_map[location]
    age=particle_age[pnum,:]

    offset=rand()*0.2
    for Ctype in Ctype_key:
        if Ctype=='CO2':
            continue
        elif Ctype=='microbe':
            plot(age,ma.masked_array(location,mask=~(form==Ctype_key[Ctype]))+offset,marker='s',ls='',ms=0.5,c=colors[Ctype],alpha=0.5,zorder=0)
        else:
            plot(age,ma.masked_array(location,mask=~(form==Ctype_key[Ctype]))+offset,marker='.',ls='',ms=0.5,c=colors[Ctype])


    transformations=nonzero(diff(form))[0]
    for trans in transformations:
        if form[trans+1]==Ctype_key['CO2']:
            scatter(age[trans],location[trans],marker='x',s=30.0,facecolor=colors[Ctype_key_inv[form[trans]]],zorder=100)
        else:
            scatter(age[trans],location[trans],marker='o',s=20.0,edgecolor=colors[Ctype_key_inv[form[trans]]],facecolor='None',zorder=100)


boundaries=nonzero(diff(location_map)!=0)[0]
poresizes=['Macropore','Micropore','Nanopore']
text(age[-1],0+2.5,poresizes[location_map[0]],rotation=90,va='bottom')
for num in boundaries:
    plot(age,zeros(len(age))+num+0.5,'k--',lw=2)
    text(age[-1],num+2.5,poresizes[location_map[num+1]],rotation=90,va='bottom')
for num in range(len(location_map)):
    plot(age,zeros(len(age))+num+0.5,'k--',lw=0.2)

Ctypes=['lignin','insoluble polymer','soluble polymer','monomer','microbe']
legend([Line2D([0],[0],ls='',marker='.',c=colors[Ctype]) for Ctype in Ctypes],Ctypes,loc=(0.0,1.06),ncol=3)

xlabel('Time')
ylabel('Pore location')
title('History of particle location')

tight_layout()


histogramfig=figure('Histograms',figsize=(8.5,8))
histogramfig.clf()
subplot(211)
bottom=zeros(len(form))
for Ctype in ['microbe','lignin','insoluble polymer','soluble polymer','monomer','CO2']:
    top=bottom+(particle_form==Ctype_key[Ctype]).sum(axis=0)/particle_form.count(axis=0)
    fill_between(age,bottom,top,label=Ctype,color=colors[Ctype])
    bottom=top

legend(ncol=2)
xlabel('Time')
ylabel('Relative amount')
title('Relative total amount of each C type over time')


subplot(212)
bottom=zeros(len(location))
location_type=ma.masked_array(location_map[particle_location],mask=particle_location.mask)
old_bottom=bottom[-1]
for poresize in ['macropore','micropore','nanopore']:
    for Ctype in ['lignin','insoluble polymer','soluble polymer','monomer','microbe']:
        top=bottom+((particle_form==Ctype_key[Ctype])&(location_type==pore_key[poresize])).sum(axis=0)/particle_form.count(axis=0)
        fill_between(age,bottom,top,label=Ctype,color=colors[Ctype])
        bottom=top
    plot(age,top,'k-',lw=2.0)
    text(age[-1],old_bottom,poresize[:-4],rotation=60,va='bottom')
    old_bottom=bottom[-1]


xlabel('Time')
ylabel('Relative amount')
title('Relative amount of each particle type in each pore size')

tight_layout()


show()
