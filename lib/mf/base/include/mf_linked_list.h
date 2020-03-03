#ifndef _SLD_MF_LINKED_LIST_H_
#define _SLD_MF_LINKED_LIST_H_

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			template<class T>
			class list
			{
			protected:
				typedef struct _node_t
				{
					_node_t* prev;
					_node_t* next;
					T item;
					_node_t(void)
						: prev(NULL)
						, next(NULL)
						, item(NULL)
					{}

					_node_t(T item)
						: prev(NULL)
						, next(NULL)
					{
						this->item = item;
					}

					T get_item(void)
					{
						return item;
					}
				} node_t;

			public:
				class position
				{
					friend class sld::lib::mf::list<T>;
				public:
					position(void)
						: _node(NULL)
					{
					}
					BOOL operator==(const position& p) const
					{
						return _node == p._node;
					}

					BOOL operator!=(const position& p) const
					{
						return _node != p._node;
					}

				private:
					const sld::lib::mf::list<T>::node_t* _node;
					position(sld::lib::mf::list<T>::node_t* p)
						: _node(p)
					{}
				};

			protected:
				typename sld::lib::mf::list<T>::node_t _anchor;
				DWORD _count;

				sld::lib::mf::list<T>::node_t* front(void) const
				{
					return _anchor.next;
				}

				sld::lib::mf::list<T>::node_t* back(void) const
				{
					return _anchor.prev;
				}

				virtual HRESULT insert_after(T item, sld::lib::mf::list<T>::node_t* before)
				{
					if (before == NULL)
						return E_POINTER;

					typename sld::lib::mf::list<T>::node_t* node = new (std::nothrow)typename sld::lib::mf::list<T>::node_t(item);
					node_t* node = new (std::nothrow)node_t(item);
					if (node == NULL)
						return E_OUTOFMEMORY;

					typename sld::lib::mf::list<T>::node_t* after = before->next;
					before->next = node;
					after->prev = node;
					node->prev = before;
					node->next = after;
					_count++;
					return S_OK;
				}

				virtual HRESULT get_item(const sld::lib::mf::list<T>::node_t* node, T* ppitem)
				{
					if (node == NULL || ppitem == NULL)
						return E_POINTER;

					*ppitem = node->item;
					return S_OK;
				}

				virtual HRESULT remove_item(sld::lib::mf::list<T>::node_t* node, T* ppitem)
				{
					if (node == NULL)
						return E_POINTER;

					assert(node != &_anchor);

					if (node == &_anchor)
						return E_INVALIDARG;

					T item;
					node->next->prev = node->prev;
					node->prev->next = node->next;

					item = node->item;
					delete node;

					_count--;

					if (ppitem)
						*ppitem = item;
					return S_OK;
				}

			public:
				list(void)
				{
					_anchor.next = &_anchor;
					_anchor.prev = &_anchor;
					_count = 0;
				}
				virtual ~list(void)
				{
					clear();
				}

				HRESULT insert_back(T item)
				{
					return insert_after(item, _anchor.prev);
				}

				HRESULT insert_front(T item)
				{
					return insert_after(item, &_anchor);
				}

				HRESULT remove_back(T* ppitem)
				{
					if (empty())
						return E_FAIL;
					else
						return remove_item(back(), ppitem);
				}

				HRESULT remove_front(T* ppitem)
				{
					if (empty())
						return E_FAIL;
					else
						return remove_item(front(), ppitem);
				}

				HRESULT get_back(T* ppitem)
				{
					if (empty())
						return E_FAIL;
					else
						return get_item(back(), ppitem);
				}

				HRESULT get_front(T* ppitem)
				{
					if (empty())
						return E_FAIL;
					else
						return get_item(front(), ppitem);
				}

				DWORD get_count(void) const
				{
					return _count;
				}

				BOOL empty(void) const
				{
					return (get_count() == 0);
				}

				template<class FN> 
				void clear(FN & clear_fn)
				{
					typename sld::lib::mf::list<T>::node_t* n = _anchor.next;
					while (n != &_anchor)
					{
						clear_fn(n->item);
						typename sld::lib::mf::list<T>::node_t* tmp = n->next;
						delete n;
						n = tmp;
					}
					_anchor.next = &_anchor;
					_anchor.prev = &_anchor;
					_count = 0;
				}

				virtual void clear(void)
				{
					clear<noop<T>>(noop<T>());
				}

				sld::lib::mf::list<T>::position front_position(void)
				{
					if (empty())
						return sld::lib::mf::list<T>::position(NULL);
					else
						return sld::lib::mf::list<T>::position(front());
				}

				sld::lib::mf::list<T>::position end_position(void) const
				{
					return sld::lib::mf::list<T>::position();
				}

				HRESULT get_item_pos(sld::lib::mf::list<T>::position pos, T* ppitem)
				{
					if (pos._node)
						return get_item(pos._node, ppitem);
					else
						return E_FAIL;
				}

				sld::lib::mf::list<T>::position Next(const sld::lib::mf::list<T>::position pos)
				{
					if (pos._node && (pos._node->next != &_anchor))
						return sld::lib::mf::list<T>::position(pos._node->next);
					else
						return sld::lib::mf::list<T>::position(NULL);
				}

				HRESULT remove(sld::lib::mf::list<T>::position& pos, T* ppitem)
				{
					if (pos._node)
					{
						typename sld::lib::mf::list<T>::node_t* node = const_cast<typename sld::lib::mf::list<T>::node_t*>(pos._node);
						pos = sld::lib::mf::list<T>::position();
						return remove_item(node, ppitem);
					}
					else
					{
						return E_INVALIDARG;
					}
				}
			};

			template<class T, BOOL NULLABLE = FALSE>
			class com_ptr_list : public list<T>
			{
			public:
				typedef T* ptr;
				void clear(void)
				{
					list<T>::clear(com_auto_release());
				}
				~com_ptr_list(void)
				{
					clear();
				}

			protected:
				HRESULT insert_after(ptr item, sld::lib::mf::list<T>::node_t* before)
				{
					if (!item && !NULLABLE)
						return E_POINTER;

					if (item)
						item->AddRef();

					HRESULT hr = list<T>::insert_after(item, before);
					if (FAILED(hr))
						sld::lib::mf::safe_release(item);

					return hr;
				}

				HRESULT get_item(const sld::lib::mf::list<T>::node_t* node, ptr* ppitem)
				{
					ptr pitem = NULL;
					HRESULT hr = list<T>::get_item(node, ppitem);
					if (SUCCEEDED(hr))
					{
						assert(pitem || NULLABLE);
						if (pitem)
						{
							*ppitem = pitem;
							(*ppitem)->AddRef();
						}

					}
					return hr;
				}

				HRESULT remove_item(sld::lib::mf::list<T>::node_t* node, ptr* ppitem)
				{
					ptr pitem = NULL;
					HRESULT hr = list<T>::remove_item(node, &pitem);
					if (SUCCEEDED(hr))
					{
						assert(pitem || NULLABLE);
						if (ppitem && pitem)
						{
							*ppitem = pitem;
							(*ppitem)->AddRef();
						}
						sld::lib::mf::safe_release(pitem);
					}
					return hr;
				}
			};

			template <class T, BOOL NULLABLE = FALSE>
			class com_ptr_list2
			{
			protected:
				typedef T* Ptr;

				typedef struct _node_t
				{
					_node_t *	prev;
					_node_t *	next;
					Ptr			item;
					_node_t(void)
						: prev(NULL)
						, next(NULL)
						, item(NULL)
					{
					}

					_node_t(Ptr item)
						: prev(NULL)
						, next(NULL)
					{
						this->item = item;
						if (item)
							item->AddRef();
					}

					~_node_t(void)
					{
						if (item)
							item->Release();
					}

					Ptr get_item(void) const 
					{ 
						return item; 
					}
				} node_t;

			public:
				// Object for enumerating the list.
				class position
				{
					friend class com_ptr_list2<T>;
				public:
					position(void)
						: _node(NULL)
					{
					}

					BOOL operator==(const position & p) const
					{
						return _node == p._node;
					}

					BOOL operator!=(const position & p) const
					{
						return _node != p._node;
					}

				private:
					const node_t * _node;

					position(node_t * p) : _node(p)
					{
					}
				};

			protected:
				node_t	_anchor;  // Anchor node for the linked list.
				DWORD	_count;   // Number of items in the list.

				typename sld::lib::mf::com_ptr_list2<T>::node_t * front(void) const
				{
					return _anchor.next;
				}

				typename sld::lib::mf::com_ptr_list2<T>::node_t * back(void) const
				{
					return _anchor.prev;
				}

				virtual HRESULT insert_after(Ptr item, typename sld::lib::mf::com_ptr_list2<T>::node_t * before)
				{
					if (before == NULL)
						return E_POINTER;

					// Do not allow NULL item pointers unless NULLABLE is true.
					if (!item && !NULLABLE)
						return E_POINTER;

					typename sld::lib::mf::com_ptr_list2<T>::node_t * node = new typename sld::lib::mf::com_ptr_list2<T>::node_t(item);
					if (node == NULL)
						return E_OUTOFMEMORY;

					typename sld::lib::mf::com_ptr_list2<T>::node_t * after = before->next;

					before->next = node;
					after->prev = node;

					node->prev = before;
					node->next = after;

					_count++;
					return S_OK;
				}

				virtual HRESULT get_item(const typename sld::lib::mf::com_ptr_list2<T>::node_t * node, Ptr * ppitem)
				{
					if (node == NULL || ppitem == NULL)
						return E_POINTER;

					*ppitem = node->item;
					if (*ppitem)
						(*ppitem)->AddRef();

					return S_OK;
				}

				// RemoveItem:
				// Removes a node and optionally returns the item.
				// ppItem can be NULL.
				virtual HRESULT remove_item(typename sld::lib::mf::com_ptr_list2<T>::node_t * node, Ptr * ppitem)
				{
					if (node == NULL)
						return E_POINTER;

					assert(node != &_anchor); // We should never try to remove the anchor node.
					if (node == &_anchor)
						return E_INVALIDARG;

					Ptr item;
					// The next node's previous is this node's previous.
					node->next->prev = node->prev;
					// The previous node's next is this node's next.
					node->prev->next = node->next;

					item = node->item;
					if (ppitem)
					{
						*ppitem = item;
						if (*ppitem)
							(*ppitem)->AddRef();
					}

					delete node;
					_count--;
					return S_OK;
				}

			public:
				com_ptr_list2(void)
				{
					_anchor.next = &_anchor;
					_anchor.prev = &_anchor;
					_count = 0;
				}

				virtual ~com_ptr_list2(void)
				{
					clear();
				}

				void clear(void)
				{
					typename sld::lib::mf::com_ptr_list2<T>::node_t * n = _anchor.next;

					// Delete the nodes
					while (n != &_anchor)
					{
						if (n->item)
						{
							n->item->Release();
							n->item = NULL;
						}

						typename sld::lib::mf::com_ptr_list2<T>::node_t * tmp = n->next;
						delete n;
						n = tmp;
					}

					// Reset the anchor to point at itself
					_anchor.next = &_anchor;
					_anchor.prev = &_anchor;
					_count = 0;
				}

				// Insertion functions
				HRESULT insert_back(Ptr item)
				{
					return insert_after(item, _anchor.prev);
				}


				HRESULT insert_front(Ptr item)
				{
					return insert_after(item, &_anchor);
				}

				// RemoveBack: Removes the tail of the list and returns the value.
				// ppItem can be NULL if you don't want the item back.
				HRESULT remove_back(Ptr * ppitem)
				{
					if (is_empty())
						return E_FAIL;
					else
						return remove_item(back(), ppitem);
				}

				// RemoveFront: Removes the head of the list and returns the value.
				// ppItem can be NULL if you don't want the item back.
				HRESULT remove_front(Ptr * ppitem)
				{
					if (is_empty())
						return E_FAIL;
					else
						return remove_item(front(), ppitem);
				}

				// GetBack: Gets the tail item.
				HRESULT get_back(Ptr * ppitem)
				{
					if (is_empty())
						return E_FAIL;
					else
						return get_item(back(), ppitem);
				}

				// GetFront: Gets the front item.
				HRESULT get_front(Ptr * ppitem)
				{
					if (is_empty())
						return E_FAIL;
					else
						return get_item(front(), ppitem);
				}

				// GetCount: Returns the number of items in the list.
				DWORD get_count(void) const 
				{ 
					return _count; 
				}

				BOOL is_empty(void) const
				{
					return (get_count() == 0);
				}

				// Enumerator functions

				typename sld::lib::mf::com_ptr_list2<T>::position front_position(void)
				{
					if (is_empty())
					{
						return sld::lib::mf::com_ptr_list2<T>::position(NULL);
					}
					else
					{
						return sld::lib::mf::com_ptr_list2<T>::position(front());
					}
				}

				typename sld::lib::mf::com_ptr_list2<T>::position end_position(void) const
				{
					return sld::lib::mf::com_ptr_list2<T>::position();
				}

				HRESULT get_item_by_position(typename sld::lib::mf::com_ptr_list2<T>::position pos, Ptr * ppitem)
				{
					if (pos)
						return GetItem(pos._node, ppitem);
					else
						return E_FAIL;
				}

				typename sld::lib::mf::com_ptr_list2<T>::position next(const typename sld::lib::mf::com_ptr_list2<T>::position pos)
				{
					if (pos._node && (pos._node->next != &_anchor))
						return sld::lib::mf::com_ptr_list2<T>::position(pos._node->next);
					else
						return sld::lib::mf::com_ptr_list2<T>::position(NULL);
				}
			};
		};
	};
};

#endif