from pylab import *

Ctype_key={'lignin':5,'insoluble polymer':4,'soluble polymer':3,'monomer':2,'microbe':1,'CO2':0}

pore_distribution={'macropore':0.3,'micropore':0.4,'nanopore':0.3}
pore_key={'macropore':0,'micropore':1,'nanopore':2}

nlocations=100

location_map=zeros(nlocations)

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
            if rand()<prob[key]:
                print('Transformation!',current_type,Ctype_key[key])
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
        prob['nanopore']=0.0
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
        prob['micropore']=0.05
        prob['nanopore']=0.0


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
    particle_age[:total_particles,tt]=particle_age[:total_particles,tt-1]+1
    for pnumber in range(total_particles):
        particle_form[pnumber,tt]=transform_particle(pnumber,tt-1)
        particle_location[pnumber,tt]=move_particle(pnumber,tt-1)
    # if tt%700==0:
    #     add_particle(total_particles,tt)
    #     total_particles += 1


histfig=figure('Particle histories')
histfig.clf()

subplot(311)
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


subplot(312)
bottom=zeros(len(form))
colors={'lignin':'C0','insoluble polymer':'C1','soluble polymer':'C2','monomer':'C3','microbe':'C4','CO2':'C5'}
for Ctype in ['lignin','insoluble polymer','soluble polymer','monomer','microbe','CO2']:
    top=bottom+(particle_form==Ctype_key[Ctype]).sum(axis=0)/particle_form.count(axis=0)
    fill_between(age,bottom,top,label=Ctype)
    bottom=top

legend()
xlabel('Time')
ylabel('Relative amount')



subplot(313)
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

tight_layout()
show()
