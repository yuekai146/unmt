import torch
import torchnn.functional as F


def l_auto(log_probs, ref_ids, ref_mask):
    '''
	log_probs: A list with each element being a FloatTensor
               (batch_size * (vocab_size+num_special_tokens)).
               len(log_probs) = l1
    ref_ids: LongTensor(batch_size * l1)
    ref_mask: FloatTensor(batch_size * l1)
    '''
    batch_size, _ = ref_ids.size()
    ref_ids = ref_ids.cpu()
    ref_probs = []
    for log_prob, ref_id in zip(
            torch.split(log_probs, 1, 1), torch.split(ref_ids, 1, 1)
            ):
    	r = torch.arange(batch_size)
    	ref_probs.append(log_prob[r.unsqueeze(-1), ref_id])

    ref_probs = torch.cat(ref_probs, dim=1)
    ref_probs = ref_probs * ref_mask
    loss = -torch.sum(ref_probs) / batch_size

    return loss


def l_cd(log_probs, ref_ids, ref_mask):
    return l_auto(log_probs, ref_ids, ref_mask)


def l_dis(scores, mask, smooth, lan):
	'''
	scores: FloatTensor (batch_size * l)
	mask: FloatTensor (batch_size * l)
	smooth: The smoothing coefficient
	lan: String ('src' or 'trg')
	'''
	batch_size = scores.size(0)
	if lan == 'src':
		loss = smooth * torch.log(scores) + \
			   (1.0 - smooth) * torch.log(1.0 - scores)
	elif lan == 'trg':
		loss = torch.log(1.0 - scores)
	else:
		raise("Invalid language type for discriminator!")

        loss = loss * mask
        loss = -torch.sum(loss)
	loss = loss / batch_size

	return loss



def l_adv(scores, mask, lan):
	'''
	scores: FloatTensor (batch_size * l)
	mask: FloatTensor (batch_size * l)
	lan: String ('src' or 'trg')
	'''
	batch_size = scores.size(0)
	if lan == 'src':
		loss = torch.log(1.0 - scores)
	elif lan == 'trg':
		loss = torch.log(scores)
	else:
		raise("Invalid language type for discriminator!")
        
        loss = loss * mask
        loss = -torch.sum(loss)
	loss = loss / batch_size

	return loss    
