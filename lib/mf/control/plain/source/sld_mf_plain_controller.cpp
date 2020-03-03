#include "sld_mf_plain_controller.h"
#include "mf_plain_controller.h"

sld::lib::mf::control::plain::controller::controller(void)
{
	_core = new (std::nothrow) sld::lib::mf::control::plain::controller::core();
}

sld::lib::mf::control::plain::controller::~controller(void)
{
	if (_core)
		delete _core;
	_core = nullptr;
}	

int32_t sld::lib::mf::control::plain::controller::open(sld::lib::mf::control::plain::controller::context_t* context)
{
	return _core->open(context);
}

int32_t sld::lib::mf::control::plain::controller::close(void)
{
	return _core->close();
}

int32_t sld::lib::mf::control::plain::controller::play(void)
{
	return _core->play();
}

int32_t sld::lib::mf::control::plain::controller::pause(void)
{
	return _core->pause();
}

int32_t sld::lib::mf::control::plain::controller::stop(void)
{
	return _core->stop();
}

int32_t sld::lib::mf::control::plain::controller::state(void) const
{
	return _core->state();
}