from pylab import *

Ctype_key={'lignin':0,'insoluble polymer':1,'soluble polymer':2,'monomer':3,'microbe':4,'CO2':5}

pore_distribution={'macropore':0.3,'micropore':0.4,'nanopore':0.3}
pore_key={'macropore':0,'micropore':1,'nanopore':2}

nlocations=100

location_map=zeros(nlocations)

location_map[0:int(pore_distribution['macropore']*nlocations)]=pore_key['macropore']
location_map[int(pore_distribution['macropore']*nlocations):int((pore_distribution['micropore']+pore_distribution['macropore'])*nlocations)]=pore_key['micropore']
location_map[int((pore_distribution['micropore']+pore_distribution['macropore'])*nlocations):]=pore_key['nanopore']


nparticles=1000
ntimes=5000
particle_age=zeros((nparticles,ntimes),dtype=int)
particle_form=zeros((nparticles,ntimes),dtype=int)
particle_location=zeros((nparticles,ntimes),dtype=int)

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
    if current_type == Ctype_key['insoluble polymer']:
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

prob_leaving={pore_key['macropore']:0.5,pore_key['micropore']:0.1,pore_key['nanopore']:0.01}

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
        prob['nanopore']=0.1
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

# Iterate model
for tt in range(1,ntimes):
    if tt%100==0:
        print(tt)
    particle_age[:total_particles,tt]=particle_age[:total_particles,tt-1]+1
    for pnumber in range(total_particles):
        particle_form[pnumber,tt]=transform_particle(pnumber,tt-1)
        particle_location[pnumber,tt]=move_particle(pnumber,tt-1)
    if tt%700==0:
        add_particle(total_particles,tt)
        total_particles += 1