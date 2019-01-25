from pylab import *

Ctype_key={'lignin':5,'insoluble polymer':4,'soluble polymer':3,'monomer':2,'microbe':1,'CO2':0}
Ctype_key_inv=Ctype_key.__class__(map(reversed,Ctype_key.items()))
colors={'lignin':'C0','insoluble polymer':'C1','soluble polymer':'C2','monomer':'C3','microbe':'C4','CO2':'C5'}

pore_distribution={'macropore':0.3,'micropore':0.4,'nanopore':0.3}
pore_key={'macropore':0,'micropore':1,'nanopore':2}

nlocations=100

location_map=zeros(nlocations,dtype=int)

location_map[0:int(pore_distribution['macropore']*nlocations)]=pore_key['macropore']
location_map[int(pore_distribution['macropore']*nlocations):int((pore_distribution['micropore']+pore_distribution['macropore'])*nlocations)]=pore_key['micropore']
location_map[int((pore_distribution['micropore']+pore_distribution['macropore'])*nlocations):]=pore_key['nanopore']


nparticles=50
ntimes=5000
particle_age=ma.masked_all((nparticles,ntimes),dtype=int)
particle_form=ma.masked_all((nparticles,ntimes),dtype=int)
particle_location=ma.masked_all((nparticles,ntimes),dtype=int)

def add_particle(pnumber,time,pore_type='macropore',C_type='insoluble polymer'):
    locs=nonzero(location_map==pore_key[pore_type])[0]
    particle_location[pnumber,time]=locs[randint(len(locs))]
    particle_age[pnumber,time]=0
    particle_form[pnumber,time]=Ctype_key[C_type]


def transform_particle(pnumber,time):
    current_type=particle_form[pnumber,time]
    current_location=particle_location[pnumber,time]
    others_in_pore=(particle_location[:,time]==current_location)
    prob={}
    microbes=sum(particle_form[others_in_pore,time]==Ctype_key['microbe'])
    if current_type == Ctype_key['lignin']:
        prob['soluble polymer']=1e-2*microbes
    if current_type == Ctype_key['insoluble polymer']:
        prob['soluble polymer']=2e-2*microbes
        prob['monomer']=2e-2*microbes
    if current_type == Ctype_key['soluble polymer']:
        prob['monomer']=2e-2*microbes
    if  current_type == Ctype_key['monomer']:
        prob['CO2']=1e-1*microbes

    nprobs=len(prob.keys())

    if nprobs>=1:
        for key in prob.keys():
            x=rand()
            if x<prob[key]:
                print('Transformation! {old:s} --> {new:s}, prob={prob:1.2f}, pore={pore:d}'.format(old=Ctype_key_inv[current_type],new=key,prob=prob[key],pore=current_location))
                return Ctype_key[key]

    return current_type

prob_leaving={pore_key['macropore']:0.1,pore_key['micropore']:0.05,pore_key['nanopore']:0.01}

def move_particle(pnumber,time):
    current_location=particle_location[pnumber,time]
    current_location_type=location_map[current_location]
    current_type=particle_form[pnumber,time]
    prob={}

    if current_type == Ctype_key['lignin']:
        prob['macropore']=0.1
        prob['micropore']=0.05
        prob['nanopore']=0.01
    if current_type == Ctype_key['insoluble polymer']:
        prob['macropore']=0.1
        prob['micropore']=0.05
        prob['nanopore']=0.01
    if current_type == Ctype_key['soluble polymer']:
        prob['macropore']=0.1
        prob['micropore']=0.05
        prob['nanopore']=0.05
    if  current_type == Ctype_key['monomer']:
        prob['macropore']=0.1
        prob['micropore']=0.1
        prob['nanopore']=0.1
    if  current_type == Ctype_key['microbe']:
        prob['macropore']=0.1
        prob['micropore']=0.01
        prob['nanopore']=0.001


    for key in prob.keys():
        x=rand()
        # print(x,prob[key]*prob_leaving[current_location_type]*pore_distribution[key])
        if x<prob[key]*prob_leaving[current_location_type]*pore_distribution[key]:
            locs=nonzero(location_map==pore_key[key])[0]
            new_location=locs[randint(len(locs))]
            # print('Movement!!',current_location,new_location)
            return new_location

    return current_location


# Initial conditions
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
    for pnumber in range(total_particles):
        if particle_form[pnumber,tt-1] == Ctype_key['CO2']:
            particle_form[pnumber,tt]=particle_form[pnumber,tt-1]
            particle_location[pnumber,tt]=particle_location[pnumber,tt-1]
            particle_age[pnumber,tt]=particle_age[pnumber,tt-1]+1
            continue
        particle_form[pnumber,tt]=transform_particle(pnumber,tt-1)
        particle_location[pnumber,tt]=move_particle(pnumber,tt-1)
        particle_age[pnumber,tt]=particle_age[pnumber,tt-1]+1



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
