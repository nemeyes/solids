#ifndef _SLD_MF_TINY_MAP_H_
#define _SLD_MF_TINY_MAP_H_

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			template <class Key, class Value>
			struct pair
			{
				Key key;
				Value value;
				pair(void) 
				{

				}
				pair(Key k, Value v)
				{
					key = k;
					value = v;
				}
			};

			template <class Key, class Value>
			class tiny_map : public sld::lib::mf::list<sld::lib::mf::pair<Key, Value>>
			{
			protected:
				typedef sld::lib::mf::pair<Key, Value> pair_type;
			public:
				tiny_map(void)
				{
					clear();
				}
				virtual ~tiny_map(void)
				{
				}

				HRESULT insert(Key k, Value v)
				{
					HRESULT hr = S_OK;

					typename sld::lib::mf::list<pair_type>::node_t * node = sld::lib::mf::list<pair_type>::front();
					while (TRUE)
					{
						if (node == &sld::lib::mf::list<pair_type>::_anchor)
						{
							hr = insert_back(pair_type(k, v));
							break;
						}
						else if (node->item.key == k)
						{
							hr = MF_E_INVALID_KEY;
							break;
						}
						else if (node->item.key > k)
						{
							hr = insert_after(pair_type(k, v), node->prev);
							break;
						}
						node = node->next;
					}
					return hr;
				}


				HRESULT remove(Key k)
				{
					HRESULT hr = E_FAIL;

					typename sld::lib::mf::list<pair_type>::node_t * node = sld::lib::mf::list<pair_type>::front();
					typename sld::lib::mf::list<pair_type>::node_t * to_remove = NULL;

					while (TRUE)
					{
						if (node == &sld::lib::mf::list<pair_type>::_anchor)
						{
							break;
						}
						else if (node->item.key == k)
						{
							to_remove = node;
							break;
						}
						else if (node->item.key > k)
						{
							hr = MF_E_INVALID_KEY;
							break;
						}
						node = node->next;
					}

					if (to_remove)
					{
						hr = remove_item(to_remove, NULL);
					}
					return hr;
				}

				HRESULT find(Key k, Value *pv)
				{
					HRESULT hr = S_OK;
					BOOL bFound = FALSE;

					pair_type pair;

					typename sld::lib::mf::list<pair_type>::position pos = sld::lib::mf::list<pair_type>::front_position();

					while (pos != sld::lib::mf::list<pair_type>::end_position())
					{
						hr = get_item_pos(pos, &pair);
						if (FAILED(hr))
						{
							break;
						}

						if (pair.key == k)
						{
							if (pv)
							{
								*pv = pair.value;
							}
							bFound = TRUE;
							break;
						}

						if (pair.key > k)
						{
							break;
						}

						pos = sld::lib::mf::list<pair_type>::next(pos);
					}
					return (bFound ? S_OK : MF_E_INVALID_KEY);
				}

				void clear()
				{
					sld::lib::mf::list<pair_type>::clear();
				}

				template <class FN>
				void clear_values(FN & clear_fn)
				{
					typename sld::lib::mf::list<pair_type>::node_t * n = sld::lib::mf::list<pair_type>::_anchor.next;
					
					while (n != &sld::lib::mf::list<pair_type>::_anchor)
					{
						clear_fn(n->item.value);

						typename sld::lib::mf::list<pair_type>::node_t* tmp = n->next;
						delete n;
						n = tmp;
					}

					sld::lib::mf::list<pair_type>::_anchor.next = &sld::lib::mf::list<pair_type>::_anchor;
					sld::lib::mf::list<pair_type>::_anchor.prev = &sld::lib::mf::list<pair_type>::_anchor;

					sld::lib::mf::list<pair_type>::_count = 0;
				}

				DWORD get_count(void) const
				{
					return sld::lib::mf::list<pair_type>::get_count();
				}

				class map_position
				{
					friend class sld::lib::mf::tiny_map;
				public:
					map_position(void)
					{
					}

					BOOL operator==(const map_position & p) const
					{
						return _pos == p.pos;
					}

					BOOL operator!=(const map_position & p) const
					{
						return _pos != p.pos;
					}

				private:
					typename sld::lib::mf::list<pair_type>::position _pos;

					map_position(typename sld::lib::mf::list<pair_type>::position p)
						: _pos(p)
					{
					}
				};


				sld::lib::mf::tiny_map<Key, Value>::map_position front_position(void)
				{
					return sld::lib::mf::tiny_map<Key, Value>::map_position(sld::lib::mf::list<pair_type>::front_position());
				}

				sld::lib::mf::tiny_map<Key, Value>::map_position end_position(void) const
				{
					return sld::lib::mf::tiny_map<Key, Value>::map_position(sld::lib::mf::list<pair_type>::end_position());
				}

				HRESULT get_value(sld::lib::mf::tiny_map<Key, Value>::map_position vals, Value * ppitem)
				{
					HRESULT hr = S_OK;
					pair_type pair;
					hr = sld::lib::mf::list<pair_type>::get_item_pos(vals.pos, &pair);

					if (SUCCEEDED(hr))
					{
						*ppitem = pair.value;
					}
					return hr;
				}

				HRESULT get_key(sld::lib::mf::tiny_map<Key, Value>::map_position vals, Key * ppitem)
				{
					HRESULT hr = S_OK;
					pair_type pair;
					hr = sld::lib::mf::list<pair_type>::get_item_pos(vals.pos, &pair);

					if (SUCCEEDED(hr))
					{
						*ppitem = pair.key;
					}
					return hr;
				}

				sld::lib::mf::tiny_map<Key, Value>::map_position next(const sld::lib::mf::tiny_map<Key, Value>::map_position vals)
				{
					return sld::lib::mf::tiny_map<Key, Value>::map_position(sld::lib::mf::list<pair_type>::next(vals.pos));
				}
			};
		};
	};
};

#endif