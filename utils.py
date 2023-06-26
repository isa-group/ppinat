from ppinat.bot.types import Literal

def create_search_grid(grid_spec):
    processed_matching = {}
    model_comb = [(x, y, z, t) for x in grid_spec["is"] for y in grid_spec["emb"] for z in grid_spec["bart"]  for t in grid_spec["vec"] if x + y + z + t == 1]
    for (x, y, z, t) in model_comb:
        for complete in grid_spec["complete"]:
            for att in grid_spec["att"]:
                for multi_heur in grid_spec["multi_heur"]:
                    name = f"is_{x}__emb_{y}__bart_{z}__vec_{t}__c_{complete}__att_{att}__mh_{multi_heur}"
                    processed_matching[name] = generate_weights(iss=x, emb=y, bart=z, vec=t, att=att, complete=complete, multi_heur=multi_heur)

    return processed_matching    

def generate_weights(iss=0, emb=0, bart=0, vec=0, att=0, complete=0, multi_heur=0):
    one_slot = {
        "slot_sim": vec * (1-att) * (1 - complete),
        "slot_complete_sim": vec * (1-att) * complete,
        "slot_is_sim": iss * (1-att) * (1 - complete),
        "slot_complete_is_sim": iss * (1-att) * complete,
        "slot_emb": emb * (1-att) * (1 - complete),
        "slot_complete_emb": emb * (1-att) * (complete),
        "bart_large_mnli_personalized": bart * (1-att) * (1-complete),
        "bart_large_mnli_personalized_complete": bart * (1-att) * complete,
        "att_is_sim": att * (1 - complete),
        "att_complete_is_sim": att * (complete)
    }

    multi_slot = {
        **({f"ev1_${k}": v/2*(1-multi_heur) for k,v in one_slot.items()}),
        **({f"ev2_${k}": v/2*(1-multi_heur) for k,v in one_slot.items()}),
        "same_type": multi_heur / 2,
        "condition_ratio": multi_heur / 2
    }

    return {
        "one_slot": one_slot,
        "multi_slot": multi_slot
    }

def print_values(values):
    for k, v in values.items():
        if isinstance(v, list):
            if len(v) > 0:
                print(f"\n{k}:")
                for i in v:
                    print(i.value)
        else:
            print(f"\n{k}:")
            print(v.value)