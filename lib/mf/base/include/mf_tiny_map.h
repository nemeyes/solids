#ifndef _SLD_MF_TINY_MAP_H_
#define _SLD_MF_TINY_MAP_H_

namespace solids
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
			class tiny_map : public solids::lib::mf::list<solids::lib::mf::pair<Key, Value>>
			{
			protected:
				typedef solids::lib::mf::pair<Key, Value> pair_type;
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

					typename solids::lib::mf::list<pair_type>::node_t * node = solids::lib::mf::list<pair_type>::front();
					while (TRUE)
					{
						if (node == &solids::lib::mf::list<pair_type>::_anchor)
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

					typename solids::lib::mf::list<pair_type>::node_t * node = solids::lib::mf::list<pair_type>::front();
					typename solids::lib::mf::list<pair_type>::node_t * to_remove = NULL;

					while (TRUE)
					{
						if (node == &solids::lib::mf::list<pair_type>::_anchor)
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

					typename solids::lib::mf::list<pair_type>::position pos = solids::lib::mf::list<pair_type>::front_position();

					while (pos != solids::lib::mf::list<pair_type>::end_position())
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

						pos = solids::lib::mf::list<pair_type>::next(pos);
					}
					return (bFound ? S_OK : MF_E_INVALID_KEY);
				}

				void clear()
				{
					solids::lib::mf::list<pair_type>::clear();
				}

				template <class FN>
				void clear_values(FN & clear_fn)
				{
					typename solids::lib::mf::list<pair_type>::node_t * n = solids::lib::mf::list<pair_type>::_anchor.next;
					
					while (n != &solids::lib::mf::list<pair_type>::_anchor)
					{
						clear_fn(n->item.value);

						typename solids::lib::mf::list<pair_type>::node_t* tmp = n->next;
						delete n;
						n = tmp;
					}

					solids::lib::mf::list<pair_type>::_anchor.next = &solids::lib::mf::list<pair_type>::_anchor;
					solids::lib::mf::list<pair_type>::_anchor.prev = &solids::lib::mf::list<pair_type>::_anchor;

					solids::lib::mf::list<pair_type>::_count = 0;
				}

				DWORD get_count(void) const
				{
					return solids::lib::mf::list<pair_type>::get_count();
				}

				class map_position
				{
					friend class solids::lib::mf::tiny_map;
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
					typename solids::lib::mf::list<pair_type>::position _pos;

					map_position(typename solids::lib::mf::list<pair_type>::position p)
						: _pos(p)
					{
					}
				};


				solids::lib::mf::tiny_map<Key, Value>::map_position front_position(void)
				{
					return solids::lib::mf::tiny_map<Key, Value>::map_position(solids::lib::mf::list<pair_type>::front_position());
				}

				solids::lib::mf::tiny_map<Key, Value>::map_position end_position(void) const
				{
					return solids::lib::mf::tiny_map<Key, Value>::map_position(solids::lib::mf::list<pair_type>::end_position());
				}

				HRESULT get_value(solids::lib::mf::tiny_map<Key, Value>::map_position vals, Value * ppitem)
				{
					HRESULT hr = S_OK;
					pair_type pair;
					hr = solids::lib::mf::list<pair_type>::get_item_pos(vals.pos, &pair);

					if (SUCCEEDED(hr))
					{
						*ppitem = pair.value;
					}
					return hr;
				}

				HRESULT get_key(solids::lib::mf::tiny_map<Key, Value>::map_position vals, Key * ppitem)
				{
					HRESULT hr = S_OK;
					pair_type pair;
					hr = solids::lib::mf::list<pair_type>::get_item_pos(vals.pos, &pair);

					if (SUCCEEDED(hr))
					{
						*ppitem = pair.key;
					}
					return hr;
				}

				solids::lib::mf::tiny_map<Key, Value>::map_position next(const solids::lib::mf::tiny_map<Key, Value>::map_position vals)
				{
					return solids::lib::mf::tiny_map<Key, Value>::map_position(solids::lib::mf::list<pair_type>::next(vals.pos));
				}
			};
		};
	};
};

#endif