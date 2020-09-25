#pragma once

#if defined(EXP_SLD_NVRENDERER_LIB)
#define EXP_SLD_NVRENDERER_CLS __declspec(dllexport)
#else
#define EXP_SLD_NVRENDERER_CLS __declspec(dllimport)
#endif

#include <sld.h>

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace nvidia
			{

				class EXP_SLD_NVRENDERER_CLS renderer
					: public solids::lib::base
				{
				public:
					class core;
				public:
					typedef struct _context_t
					{
						void*		cuctx;
						int32_t		width;
						int32_t		height;
						HWND		hwnd;
						int32_t		nstaging;
						_context_t(VOID)
							: cuctx(NULL)
							, width(-1)
							, height(-1)
							, hwnd(NULL)
							, nstaging(1)
						{}
					} context_t;

					renderer(void);
					virtual ~renderer(void);

					BOOL		is_initialized(void);
					int32_t		initialize(solids::lib::video::nvidia::renderer::context_t* ctx);
					int32_t		release(void);
					int32_t		render(uint8_t* deviceptr, int32_t pitch);

				private:
					renderer(const renderer& clone);

				private:
					solids::lib::video::nvidia::renderer::core* _core;
				};

			};
		};
	};
};

