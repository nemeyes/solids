#include "sld_nvdecoder.h"
#include "nvdecoder.h"

namespace solids
{
namespace lib
{
namespace video
{
namespace nvidia
{

	decoder::decoder(void)
	{
		_core = new solids::lib::video::nvidia::decoder::core(this);
	}

	decoder::~decoder(void)
	{
		if (_core)
		{
			if (_core->is_initialized())
				_core->release();
			delete _core;
			_core = NULL;
		}
	}

	BOOL decoder::is_initialized(void)
	{
		return _core->is_initialized();
	}

	void* decoder::context(void)
	{
		return _core->context();
	}

	int32_t decoder::initialize(solids::lib::video::nvidia::decoder::context_t* ctx)
	{
		return _core->initialize(ctx);
	}

	int32_t decoder::release(void)
	{
		return _core->release();
	}

	int32_t decoder::decode(uint8_t* bitstream, int32_t bitstreamSize, long long bitstreamTimestamp, uint8_t*** decoded, int32_t* numberOfDecoded, long long** timetstamp)
	{
		return _core->decode(bitstream, bitstreamSize, bitstreamTimestamp, decoded, numberOfDecoded, timetstamp);
	}

	size_t decoder::get_pitch(void)
	{
		return _core->get_pitch();
	}

	size_t decoder::get_pitch_resized(void)
	{
		return _core->get_pitch_resized();
	}

	size_t decoder::get_pitch_converted(void)
	{
		return _core->get_pitch_converted();
	}

	size_t decoder::get_pitch2(void)
	{
		return _core->get_pitch2();
	}

};
};
};
};

